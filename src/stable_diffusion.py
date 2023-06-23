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
    PNDMScheduler
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
import clip

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), dtype=torch.float16, device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


CACHE_DIR = "/source/skg/diffusion-project/hugging_cache"
class StableDiffusion(nn.Module):
    def __init__(self, device, 
                 model_name     = 'CompVis/stable-diffusion-v1-4',
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
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id).to(self.device)
        # self.image_encoder = None
        # self.image_processor = None        
        # self.image_processor = AutoProcessor.from_pretrained(self.model_id)
        # self.clipmodel       = CLIPModel.from_pretrained(self.model_id).to(self.device)
        
        self.criterionCLIP   = torch.nn.CosineSimilarity(dim=1, eps=1e-15)
        self.criterionL1     = torch.nn.L1Loss()
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.latent_gray = torch.tensor(
                [0.9071, -0.7711,  0.7437,  0.1510]
            )[None].unsqueeze(-1).unsqueeze(-1).to(self.device)
        
        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        if concept_name is not None:
            self.load_concept(concept_name)
        logger.info(f'\t successfully loaded stable diffusion!')
        
        self.aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def img_clip_loss(self, clip_model, rgb1, rgb2):
        image_z_1 = clip_model.encode_image(self.aug(rgb1))
        image_z_2 = clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, clip_model, rgb, prompt):
        image_z_1 = clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        text = clip.tokenize(prompt).to(self.device)
        text_z = clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss


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
    
    def optimize_text_token(self, prompt, image_features, max_itr=100):
        """
        Args:
            prompt (str): initial text prompt
            image_features ()
        Return:
            text_optimized
        """
        view_text_z = []
        for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
            text = f"{d} view"
            view_text_z.append(self.diffusion.get_text_embeds([text]))
        import pdb;pdb.set_trace()
        
        with torch.no_grad():
            # txt -> token -> txt embed (hidden)
            text_input = self.tokenizer(
                prompt, 
                padding='max_length', 
                max_length=self.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.device)
            # text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt').to(self.device)
            text_embeddings = self.clipmodel.text_model(text_input.input_ids.to(self.device))[0]
            
            uncond_input = self.tokenizer(
                [''] * len(prompt), 
                padding='max_length', 
                max_length=self.tokenizer.model_max_length, 
                return_tensors='pt')

            uncond_embeddings = self.clipmodel.text_model(uncond_input.input_ids.to(self.device))[0]
            
            ## normalized image feature
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_embeddings.requires_grad = True
        
        # image  = Image.open("/source/kseo/diffusion_playground/latent-nerf-test/data/monster_ball.jpg")
        # prompt = "pokemon, monster ball, red and white color, on a grass"
        # inputs = self.image_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        # outputs = self.clipmodel(**inputs.to(self.device))
        
        itr = 0
        save_itr = max_itr // 5
        # max_itr = 100
        pbar = tqdm(total=max_itr, initial=itr)
        optimizer = torch.optim.Adam([text_embeddings], lr=1e-5, betas=(0.9, 0.99), eps=1e-15)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda epoch: 0.95 ** epoch,
        #     last_epoch=-1,
        #     verbose=False)

        text_optimized = None
        prev_loss = 100
        while itr < max_itr:
            itr += 1
            pbar.update(1)
            
            optimizer.zero_grad()            
            # txt embed (hidden) -> txt feature
            pooled_output = text_embeddings[
                torch.arange(text_embeddings.shape[0], device=self.device),
                text_input.input_ids.to(dtype=torch.int, device=self.device).argmax(dim=-1),
            ]
            text_features = self.clipmodel.text_projection(pooled_output)[0]
                
            # normalized text features
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            ## cosine similarity
            loss = 1.0 - self.criterionCLIP(text_features, image_features.detach())
            
            loss.backward()
            optimizer.step()
            
            # pbar.set_description("prev: {:06f} loss: {:06f} ".format(prev_loss, loss.item()))
            pbar.set_description("loss: {:06f} ".format(loss.item()))
            
            if loss < prev_loss:
                prev_loss      = loss.item()
                text_optimized = text_embeddings.detach().clone()
                
            if itr % save_itr == 0 or itr == max_itr:
                img = self.embeds_to_img(torch.cat([uncond_embeddings.detach(), text_embeddings]))                
                Image.fromarray(img[0]).save('test_{:03}.png'.format(itr))
                img = self.embeds_to_img(torch.cat([uncond_embeddings.detach(), text_optimized]))
                Image.fromarray(img[0]).save('best.png')
        return text_optimized

    def embeds_to_img(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        # Text embeds -> img latents
        # import pdb;pdb.set_trace()
        text_embeddings.detach()
        latents = self.produce_latents(text_embeddings, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        
        return imgs
    
    def embeds_to_img_tensor(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        # Text embeds -> img latents
        text_embeddings.detach()
        latents = self.produce_latents(text_embeddings, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents_grad(latents) # [1, 3, 512, 512]        
        return imgs

    def train_step(self, 
                   text_embeddings, 
                   inputs, 
                   guidance_scale=100):
        """
        Input:
            text_embeddings:    (n, 768), text feature after the projection
            inputs:             (n,c,h,w), rendered latent image
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

        # w(t), alpha_t * sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        # grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0 # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs
    
    def decode_latents_grad(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        imgs = self.vae.decode(latents).sample
        imgs = ((imgs * 0.5) + 0.5).clamp(0, 1)        
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




