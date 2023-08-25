from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPVisionModel,
    logging, 
    CLIPProcessor, 
    AutoProcessor,
    CLIPTextModelWithProjection, 
    CLIPVisionModelWithProjection,
    CLIPFeatureExtractor,
    CLIPModel,
)
from diffusers import (
    AutoencoderKL, 
    UNet2DModel,
    UNet2DConditionModel, 
    PNDMScheduler,
    DDIMScheduler,
    DDIMInverseScheduler
)

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from tqdm import tqdm
import time
# import clip

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), dtype=torch.float16, device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

MODEL_NAME = 'CompVis/stable-diffusion-v1-4'
CACHE_DIR = "/source/skg/diffusion-project/hugging_cache"
class StableDiffusion(nn.Module):
    def __init__(self, device, 
                 model_name     = MODEL_NAME,
                 concept_name   = None, 
                 latent_mode    = True,
                 cache_dir      = CACHE_DIR,
                 ):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        logger.info(f'loading stable diffusion with {model_name}...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.model_id = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, cache_dir=cache_dir).to(self.device)
        # self.image_encoder = None
        # self.image_processor = None        
        # self.image_processor = AutoProcessor.from_pretrained(self.model_id)
        # self.clipmodel       = CLIPModel.from_pretrained(self.model_id).to(self.device)
        
        self.criterionCLIP   = torch.nn.CosineSimilarity(dim=1, eps=1e-15)
        self.criterionL1     = torch.nn.L1Loss()
        self.criterionL2     = torch.nn.MSELoss()
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])        
        self.aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.latent_gray = torch.tensor(
                [0.9071, -0.7711,  0.7437,  0.1510]
            )[None].unsqueeze(-1).unsqueeze(-1).to(self.device)
        
        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", num_train_timesteps=self.num_train_timesteps)
        self.scheduler_inv = DDIMInverseScheduler.from_pretrained(model_name, subfolder="scheduler", num_train_timesteps=self.num_train_timesteps, prediction_type='sample')
        ### epsilon, sample, v_prediction
        
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        self.linear_rgb_estimator = torch.tensor([
            #   R       G       B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ]).to(self.device)
        
        if concept_name is not None:
            self.load_concept(concept_name)
        logger.info(f'\t successfully loaded stable diffusion!')


    def load_concept(self, concept_name):
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            # text_embeddings = self.text_encoder(text_input.input_ids.to(self.device)).text_embeds
            # text_embeddings = self.clipmodel.text_model(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            [''] * len(prompt), 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device)).text_embeds
            # uncond_embeddings = self.clipmodel.text_model(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_image_embeds(self, image_path):
        if type(image_path) == str:
            image = Image.open(image_path)
        else:
            image = image_path
            
        image_inputs = self.image_processor(
                images=image, 
                return_tensors="pt"
            )        
        with torch.no_grad():
            image_features = self.clipmodel.get_image_features(image_inputs.pixel_values.to(self.device))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features, self.transform(image)
    
    def img_clip_loss(self, clip_model, rgb1, rgb2):
        image_z_1 = clip_model.encode_image(self.aug(rgb1))
        image_z_2 = clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, clip_model, rgb, text_z):
        image_z_1 = clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss

    def embeds_to_img(self, 
                      text_embeddings, 
                      size=512, 
                      num_inference_steps=50, 
                      guidance_scale=7.5, 
                      latents=None,
                      out_tensor=False,
                      ):
        # Text embeds -> img latents
        # import pdb;pdb.set_trace()
        # with torch.no_grad():
        latents = self.produce_latents(text_embeddings, height=size, width=size, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, latents=latents) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        if out_tensor:
            return self.decode_latents_grad(latents, normalize=False) # [1, 3, 512, 512]
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        
        return imgs
        
    def train_step(self,
                   text_embeddings, 
                   inputs, 
                   ref_img_tensor=None,
                   guidance_scale=100,
                   use_clip=False,
                   clip_model=None,
                   ):
        """
        Args
            text_embeddings:    (n, 768), text feature after the projection
            inputs (torch.Tensor):          [N, 4, 64, 64], rendered latent
            ref_img_tensor (torch.Tensor):  [N, 1, 512, 512],  reference image
        """
        # interp to 512x512 to be fed into vae.
               
        # _t = time.time()
        if not self.latent_mode:
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents) # (64 x 64 x 4)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            ### UNet2DModel,
            # noise_pred = self.unet(latent_model_input, t).sample
            ### UNet2DConditionModel, 
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # if use_clip and (t / self.num_train_timesteps) <= 0.4:
        # if use_clip and (t / self.num_train_timesteps) >= 0.4:
        if use_clip:
            ### no gradient
            # self.scheduler.set_timesteps(self.num_train_timesteps)
            # de_latents = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']
            # de_latents = de_latents.detach().requires_grad_()
            
            de_latents, pred_x0 = self.step(noise_pred, t, latents)
            
            imgs = self.decode_latents_grad(de_latents)
            
            loss_img = self.img_clip_loss(clip_model, imgs, (ref_img_tensor*0.5+0.5))
            loss_txt = self.img_text_clip_loss(clip_model, imgs, text_embeddings)
            loss = 10 * (loss_img + loss_txt)
            # import pdb;pdb.set_trace()
            loss.backward(retain_graph=True)
            # transforms.ToPILImage()(imgs[0]).save('test.png')
            # transforms.ToPILImage()(ref_img_tensor[0]*0.5+0.5).save('test.png')
        else:
            ## w(t), alpha_t * sigma_t^2
            w = (1 - self.alphas[t])
            # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            # grad = grad.clamp(-1, 1)
            # grad = torch.nan_to_num(grad)

            # manually backward, since we omitted an item in grad and cannot simply autodiff.
            # _t = time.time()
            latents.backward(gradient=grad, retain_graph=True)
            loss = torch.zeros([1]).to(self.device)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss # dummy loss value

    def produce_latents(self, 
                        text_embeddings, 
                        height=512, 
                        width=512, 
                        num_inference_steps=50, 
                        guidance_scale=7.5, 
                        latents=None,
                        start=0,
                       ):
        if guidance_scale <= 1:
            text_embeddings = text_embeddings[-1][None]
            
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            # for i, t in enumerate(self.scheduler.timesteps):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sample")):
                # skip ! :: this is index not timestep
                if i <= start:
                    continue
                    
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                if guidance_scale > 1:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                
                # perform guidance
                if guidance_scale > 1:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                        
        return latents
    
    def produce_latents_guide(self, 
                              text_embeddings, 
                              height=512, 
                              width=512, 
                              num_inference_steps=50, 
                              guidance_scale=7.5, 
                              latents=None, 
                              start=0,
                              # guide_latents=None, 
                              guide_latents_list=None,
                              uncond_embeddings_list=None,
                              clip_model=None, 
                              beta=1.6,
                             ):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        max_guidance_step = self.num_train_timesteps // 2

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sample")):
                # skip !
                if i <= start:
                    continue
                    
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                with torch.no_grad():
                    latents_old = latents.clone()
                    
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    if uncond_embeddings_list != None:
                        text_embeddings[0] = uncond_embeddings_list[i]
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                
                """
                latent guidence using anti-gradient :: 
                reference: Sketch-Guided Text-to-Image Diffusion Models
                """
                # if t >= max_guidance_step:

#                 with torch.no_grad():
#                     _latents = latents.clone()

#                 _latents.requires_grad = True
#                 ## convert 4 channel to 3 channel

#                 #### instead of direct comparison w final latent, maybe it is better to use forward noise?
#                 # guide_latents = guide_latents_list[i]
#                 guide_latents = guide_latents_list[-1]
#                 est_latents = torch.einsum('bi,abcd->aicd', self.linear_rgb_estimator, _latents) * 0.5 + 0.5
#                 est_guide   = torch.einsum('bi,abcd->aicd', self.linear_rgb_estimator, guide_latents) * 0.5 + 0.5
                
#                 # loss = self.criterionL1(image_z_1, image_z_2) ## L1 dist
#                 loss = self.img_clip_loss(clip_model, est_latents, est_guide) ## cos-sim
                
#                 loss.backward()
#                 gradient = _latents.grad

#                 # gradient = latents - guide_latents
#                 # weight = self.criterionL2(latents_old, latents) / torch.norm(gradient) * beta ### paper
#                 # weight = 1.0
#                 # weight = (self.num_train_timesteps - t) / self.num_train_timesteps
#                 # weight = 1.0
#                 # weight = t / self.num_train_timesteps
#                 weight = self.num_train_timesteps / (self.num_train_timesteps - t) * beta
#                 # weight = self.num_train_timesteps
#                 # import pdb;pdb.set_trace()


#                 latents = latents - (gradient * weight)
                
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def decode_latents_grad(self, latents, normalize=True):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        imgs = self.vae.decode(latents).sample
        if normalize:
            imgs = (imgs * 0.5) + 0.5
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
   
    def null_optimization(self, 
                          latents_list: torch.Tensor,
                          text_embeddings: torch.Tensor,
                          num_inference_steps=50,
                          num_inner_steps=10,
                          guidance_scale=7.5,
                          epsilon=1e-5,
                         ):
        """
        Reference: https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
        """
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        uncond_embeddings_list = []
        
        # latent_cur = latents[-1] # ===50
        latent_cur = latents_list[0].clone()
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        # for i in range(NUM_DDIM_STEPS):
        pbar = tqdm(total=num_inner_steps * num_inference_steps, desc="null-text optimization")
        for i in range(num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            
            optimizer = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            # latent_prev = latents[len(latents) - i - 2] #=== 48, 47, 46 ... -1(49) ?
            if i+1 >= num_inference_steps:
                i = -1
            latent_prev = latents_list[i+1]
            
            t = self.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_con = self.unet(latent_cur, t, encoder_hidden_states=cond_embeddings)['sample']
                # noise_pred_con = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                
            for j in range(num_inner_steps):
                noise_pred_uncon = self.unet(latent_cur, t, encoder_hidden_states=uncond_embeddings)['sample']
                # noise_pred_uncon = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                latents_prev_rec = self.scheduler.step(noise_pred, t, latent_cur)['prev_sample']
                
                # import pdb;pdb.set_trace()
                loss = self.criterionL2(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_item = loss.item()
                pbar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
                    
            for j in range(j + 1, num_inner_steps):
                pbar.update()
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            
            with torch.no_grad():
                # import pdb;pdb.set_trace()
                embeddings = torch.cat([uncond_embeddings, cond_embeddings])
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_input = torch.cat([latent_cur] * 2)
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=embeddings)["sample"]
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                latent_cur = self.scheduler.step(noise_pred, t, latent_cur)['prev_sample']
                
        # for i in range(i + 1, num_inner_steps):
        #     pbar.update()
        pbar.close()
        return uncond_embeddings_list
    
    @torch.no_grad()
    def invert(self,
               latents: torch.Tensor,    
               text_embeddings: torch.Tensor,
               num_inference_steps=50,
               guidance_scale=7.5,
               eta=0.0,
               return_intermediates=False,
               **kwds
            ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """        
        start_latents = latents
        
        if guidance_scale <= 1.:
            # use conditino
            text_embeddings = text_embeddings[-1][None]
        print("latents shape: ", latents.shape)
        
        # interative sampling
        # self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        self.scheduler_inv.set_timesteps(num_inference_steps-1)
        print("Valid timesteps: ", self.scheduler_inv.timesteps)
        
        
        latents_list = [latents]
        # pred_x0_list = [latents]
        
        # self.diffusion.scheduler.timesteps
        # for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
        for i, t in enumerate(tqdm(self.scheduler_inv.timesteps, desc="DDIM Inversion")):
            # t = t-1
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)['sample']
            
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                
            # compute the previous noise sample x_t-1 -> x_t
            # latents_, pred_x0 = self.next_step(noise_pred, t, latents)
            # latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            latents = self.scheduler_inv.step(noise_pred, t, latents)['prev_sample']
            #### prev_timestep = 981 + self.diffusion.num_train_timesteps // 50
            
            latents_list.append(latents)
            # pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
    
if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
