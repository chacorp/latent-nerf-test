from transformers import (
    logging, 
    CLIPImageProcessor, 
    CLIPFeatureExtractor,
    CLIPTokenizer,
    CLIPTextModel
    )
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from transformers import AutoProcessor, CLIPVisionModel, CLIPImageProcessor
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import time
from torchvision import transforms
import clip

MODEL_NAME = 'Fantasy-Studio/Paint-by-Example'
CACHE_DIR = "/source/skg/diffusion-project/hugging_cache"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
class PaintbyExample(nn.Module):
    def __init__(self, device, 
                 model_name  = MODEL_NAME,
                 latent_mode = True,
                 cache_dir   = CACHE_DIR,
                 step_range  = [0.2, 0.6]
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

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        logger.info(f'loading stable diffusion with {model_name}...')
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", cache_dir=cache_dir,use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        logger.info(f'loading clip image encoder...')
        self.image_encoder = PaintByExampleImageEncoder.from_pretrained(model_name, subfolder="image_encoder", cache_dir=cache_dir).to(self.device)
        
        self.model_id = "openai/clip-vit-base-patch32"
        # self.image_processor = AutoProcessor.from_pretrained(self.model_id)
        # self.CLIP_image_encoder = CLIPVisionModel.from_pretrained(model_name, subfolder="image_encoder", cache_dir=cache_dir).to(self.device)
        # loading text processor
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id).to(self.device)

        # loading image processor
        logger.info(f'loading clip image processor...')
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(self.model_id, cache_dir=cache_dir)


        # 3. The UNet model for generating the latents.
        logger.info(f'loading unet...')
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet" \
            , cache_dir=cache_dir, use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        # self.scheduler = PNDMScheduler.from_pretrained(model_name, subfolder="scheduler", cache_dir=cache_dir, use_auth_token=self.token)
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.num_inference_steps = 50
        self.min_step = int(self.num_train_timesteps * float(step_range[0]))
        self.max_step = int(self.num_train_timesteps * float(step_range[1]))
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
       
        logger.info(f'\t successfully loaded stable diffusion!')
        self.linear_rgb_estimator = torch.tensor([
            #   R       G       B
            [ 0.298,  0.207,  0.208],  # L1
            [ 0.187,  0.286,  0.173],  # L2
            [-0.158,  0.189,  0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ]).to(self.device)
        
        self.latent_grey = torch.tensor(
            #    L1      L2       L3      L4
            [[0.9071, -0.7711,  0.7437,  0.1510]]
        ).unsqueeze(-1).unsqueeze(-1).to(self.device)
        
        self.aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def clip_image_process(self, np_img):
        return self.feature_extractor(np_img, return_tensors="pt").pixel_values.to(self.device)


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

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            [''] * len(prompt), 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

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
    
    def _encode_image(self, image, num_images_per_prompt=1):
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device)
        image_embeddings, negative_prompt_embeds = self.image_encoder(image, return_uncond_vector=True)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, image_embeddings.shape[0], 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, 1, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    def step(self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta: float=0.0,
            verbose=False,):
        """
        Args
            model_output (torch.tensor): noise
            timestep (int): t
            x (torch.tensor): latents
        """
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0
    
    def produce_latents(self,
            latents_masked_images, # [2, 4, 64, 64]
            latents_masks, # [2, 4, 64, 64]
            latents, # [1, 4, 64, 64]
            image_embeddings,
            guidance_scale=7.5):

        # from tqdm import tqdm
        # print("denoise...")
        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
            # for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # latents = [1, 4, 64, 64]
                
                latent_model_input = torch.cat([latents] * 2) 
                # -> extend in batch [2, 4, 64, 64]
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 
                # -> [2, 4, 64, 64]
                
                latent_model_input = torch.cat([latent_model_input, latents_masked_images, latents_masks], dim=1)
                # -> [2, 9, 64, 64]
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)['prev_sample']
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                
                # if i % 5==0 and 0:
                #     vis = torch.cat([
                #             latents[0],
                #             latents_masked_images[0],
                #             torch.cat([latents_masks[0], latents_masks[0], latents_masks[0], latents_masks[0]], dim=0),
                #         ], dim=1)
                #     torchvision.utils.save_image(vis, 'ttt{:03}.png'.format(i))
                #     # import pdb;pdb.set_trace()
                #     # transforms.ToPILImage()(approx).save('tt.png')
                #     # transforms.ToPILImage()(latents[0]).save('tt.png')
                #     # TODO:
                #     # approx = torch.einsum('bj,bik->jik', self.linear_rgb_estimator, latents[0])
                #     imgs = self.decode_latents(latents)
                #     imgs = self.denorm_img(imgs)
                #     transforms.ToPILImage()(imgs[0]).save('tt{:03}.png'.format(i))
    
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        # imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def decode_latents_grad(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        imgs = self.vae.decode(latents).sample

        # imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        # imgs = 2 * imgs - 1
        
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def forward(self, \
        images, \
        masks, \
        example_images, \
        num_inference_steps=50, \
        strength=0.8, \
        guidance_scale=7.5,
        latents=None):
        """
        Args
            images (torch.Tensor): [B, 3, H, W]
            masks (torch.Tensor):  [B, 1, H, W]
            example_images (torch.Tensor): [B, 3, 224, 244] --> preprocess with clip_image_process
            num_inference_steps (int): number of inference steps
            guidance_scale (float): guidance scale
            strength (float): strength of the original input images
            latents (torch.Tensor): [B, 4, H//8, W//8]
        Return
            input image range: [0, 1]
            mask range: [0, 1]
            output image range: [0, 1]
        """
        # normalize images
        images = self.normalize_img(images)
        # torch.Size([1, 3, 512, 512])
        
        # binarize mask
        masks = self.binarize_mask(masks)
        # torch.Size([1, 1, 512, 512])
        
        masks = 1 - masks
        masked_images = images * masks
        # torch.Size([1, 3, 512, 512])
        
        
        # encode images
        # import pdb;pdb.set_trace()
        latents_masked_images = self.encode_imgs(masked_images) # torch.Size([1, 4, 64, 64])
        
        bs, _, h, w = latents_masked_images.shape
        latents_masks = F.interpolate(masks, (h,w))             # torch.Size([1, 1, 64, 64])
        
        # import pdb;pdb.set_trace()

        # --------------------------------------------------------------
        # image clip embedding
        # image -> image embeds
        if False:
            img_embeds = self._encode_image(example_images)
        
        else:
            img_embeds, neg_img_embeds = self.image_encoder(
                example_images, 
                return_uncond_vector=True)
            # duplicate image embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = img_embeds.shape
            img_embeds = torch.cat([neg_img_embeds, img_embeds])
        
        
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        # timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1.0)
        # latent_timestep = timesteps[:1].repeat(bs)
        # print("[diffusion forward] num_inference_steps", num_inference_steps)
        # print("[diffusion forward] latent_timestep", latent_timestep)

        

        # set random lantent
        if latents is None:
            latents = torch.randn_like(latents_masked_images, device=latents_masked_images.device)
       
        # guided inference
        latents_masks         = torch.cat([latents_masks]*2, dim=0)
        latents_masked_images = torch.cat([latents_masked_images]*2, dim=0)
        # img_embeds            = torch.cat([img_embeds]*2, dim=0)

        # Text embeds -> img latents
        latents = self.produce_latents(
            latents_masked_images=latents_masked_images,
            latents_masks=latents_masks,
            latents=latents,
            image_embeddings=img_embeds,
            guidance_scale=guidance_scale) 
        # import pdb;pdb.set_trace()
        # transforms.ToPILImage()((latents[0]*0.5 + 0.5)).save('test.png')
        # decode
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # denormalize imgs
        imgs = self.denorm_img(imgs)
        return imgs

    def lantent_forward(self,
        latents, 
        latents_masks, 
        example_images, 
        num_inference_steps=50, 
        strength=0.8, 
        guidance_scale=7.5,
        rand_latent = False,
        ):
        """
        Args
            images (torch.Tensor): [B, 4, 64, 64]
            masks (torch.Tensor):  [B, 1, 64, 64]
            example_images (torch.Tensor): [B, 3, 224, 244] --> preprocess with clip_image_process
            num_inference_steps (int): number of inference steps
            guidance_scale (float): guidance scale
            strength (float): strength of the original input images
            latents (torch.Tensor): [B, 4, H//8, W//8]
        Return
            input image range:  [0, 1]
            mask range:         [0, 1]
            output image range: [0, 1]
        """
        latents_masks = 1 - latents_masks
        latents_masked_images = latents * latents_masks
        # latents_grey_masks    = torch.cat([latents_masks] * 4, dim=1) * self.latent_grey
        # latents_masked_images = latents_masked_images + latents_grey_masks
        latents_masked_images = latents_masked_images
        
        # --------------------------------------------------------------
        # image clip embedding
        
        example_images = self.clip_image_process(example_images)
        img_embeds, neg_img_embeds = self.image_encoder(example_images, return_uncond_vector=True)
        img_embeds = torch.cat([neg_img_embeds, img_embeds])
                
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        # self.scheduler.set_timesteps(1, device=self.device)
       
        # guided inference
        latents_masks         = torch.cat([latents_masks]*2, dim=0)
        latents_masked_images = torch.cat([latents_masked_images]*2, dim=0)
       
        # set random lantent
        if rand_latent:
            latents = torch.randn_like(latents, device=self.device)
       
        # Text embeds -> img latents
        # latents = self.produce_latents(
        #     latents_masked_images=latents_masked_images,
        #     latents_masks=latents_masks,
        #     latents=latents,
        #     image_embeddings=img_embeds,
        #     guidance_scale=guidance_scale) 
        def sample(latents_masked_images, latents_masks, latents, img_embeds, guidance_scale):
            with torch.autocast('cuda'):
                for i, t in enumerate(self.scheduler.timesteps):
                    
                    latent_model_input = torch.cat([latents] * 2) 
                    # -> extend in batch [2, 4, 64, 64]
                    
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 
                    # -> [2, 4, 64, 64]
                    
                    latent_model_input = torch.cat([latent_model_input, latents_masked_images, latents_masks], dim=1)
                    # -> [2, 9, 64, 64]
                    
                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=img_embeds)['sample']
                    # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=img_embeds)['sample']

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    latents, pred_x0 = self.step(noise_pred, t, latents)
            return latents, pred_x0
        
        latents, pred_x0 = sample(latents_masked_images, latents_masks, latents, img_embeds, guidance_scale)

        # decode
        imgs = self.decode_latents_grad(latents) # [1, 3, 512, 512]
        # imgs = torch.einsum('bj,nbik->njik', self.linear_rgb_estimator, latents) # [1, 3, 64, 64]
        
        # denormalize imgs
        imgs = imgs * 0.5 + 0.5
        # imgs = self.denorm_img(imgs)
        return imgs

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start
    
    def approx_latent2rgb(self, ref_img):
        import numpy as np
        from tqdm import tqdm
        rs_ref_img = ref_img.resize((512,512))
        np_ref_img = np.array(rs_ref_img).transpose(2,0,1)
        th_ref_img = torch.tensor(np_ref_img)[None] / 255.
        th_ref_img = th_ref_img.to(self.device)
        latents_ref_img = self.encode_imgs(th_ref_img) # torch.Size([1, 4, 64, 64])
        const = 1 / 0.18215 
        latents_ref_img = latents_ref_img[0] # torch.Size([4, 64, 64])
        # homogeneous coordinate
        
        down_ref_img = transforms.ToTensor()(ref_img.resize((64,64))) # torch.Size([1, 3, 64, 64])
        down_ref_img = down_ref_img.to(self.device) ### 0 ~ 1 range
        # down_ref_img = (down_ref_img * 2.0) - 1.0  ### -1 ~ 1 range
        
        
        latents_ref_img = torch.cat([latents_ref_img, torch.ones(1,64,64).to(self.device)])
        
        # new_mat = torch.tensor([[ 0.2135,  0.1282,  0.1321],[ 0.0707,  0.2170,  0.1689],[-0.1449,  0.1692,  0.1697],[-0.1179, -0.2380, -0.2821]], device='cuda:0', requires_grad=True)
        # new_mat = torch.tensor([[ 0.2117,  0.1266,  0.1307],[ 0.0700,  0.2151,  0.1670],[-0.1429,  0.1672,  0.1677],[-0.1161, -0.2360, -0.2801]], device='cuda:0', requires_grad=True)
        new_mat = torch.tensor([[ 0.2135,  0.1282,  0.1321],[ 0.0707,  0.2170,  0.1689],[-0.1449,  0.1692,  0.1697],[-0.1179, -0.2380, -0.2821],[ 0.2555,  0.0639,  0.0823]], device='cuda:0', requires_grad=True)
        
        def svae_image(decode_latents, norm=True):
            if norm:
                vis = torch.cat([(decode_latents*0.5) + 0.5, (down_ref_img*0.5)+0.5], dim = -1)
            else:
                vis = torch.cat([decode_latents, down_ref_img], dim = -1)
            transforms.ToPILImage()(vis).save('tt.png')
            
        def optimize_new_mat(new_mat, latent, lr=1e-3, L=1, nl=0, max_itr=2000):
            criterionL1 = nn.L1Loss()
            optimizer = torch.optim.Adam([new_mat], lr=lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            itr = 0
            pbar = tqdm(total=max_itr, initial=itr)
            print('optimize')
            while itr < max_itr:
                itr += 1
                pbar.update(1)
                optimizer.zero_grad()
                decode_latents = torch.einsum('bj,bik->jik', new_mat, latent)
                if L==1:
                    loss = criterionL1(decode_latents, down_ref_img.detach())
                elif L==2:
                    loss = F.mse_loss(decode_latents, down_ref_img.detach())
                else:
                    loss = F.mse_loss(decode_latents, down_ref_img.detach()) + criterionL1(decode_latents, down_ref_img.detach())
                if nl:
                    loss += new_mat.norm()
                loss.backward()
                optimizer.step()
                pbar.set_description("loss: {:06f} ".format(loss.item()))
                scheduler.step()
        
            svae_image(decode_latents, 0)
            return new_mat
        
        
        # import pdb; pdb.set_trace()
        # R = torch.linalg.lstsq(latents_ref_img.reshape(-1,4), down_example_images[0].reshape(-1,1)).solution
        
        # # X = torch.linalg.pinv(latents_ref_img) @ down_example_images
        # pinv_l = torch.linalg.pinv(latents_ref_img)
        # X = torch.einsum('jbk,ibk->ji', pinv_l, down_example_images)
        # image = torch.einsum('bj,bik->jik', X, latents_ref_img)
        # svae_image(torch.einsum('bj,bik->jik', X, latents_ref_img), 0)

        # R_img = torch.einsum('bj,bik->jik', R, latents_ref_img)
        # vis = torch.cat([(R_img*0.5) + 0.5, (down_example_images[2][None]*0.5)+0.5], dim = -1)
        # transforms.ToPILImage()(vis).save('tt.png')
        # svae_image(torch.einsum('bj,bik->jik', R, latents_ref_img), 0)
        
        new_mat = optimize_new_mat(new_mat, latents_ref_img, lr=1e-4, L=3, nl=1, max_itr=1000)
        # svae_image(torch.einsum('bj,bik->jik', new_mat-n_linear, latents_ref_img), 0)
        # svae_image(torch.einsum('bj,bik->jik', optimize_new_mat(new_mat, lr=1e-4, L=2, max_itr=2000), latents_ref_img), 0)
        
        # _linear = torch.cat([self.linear_rgb_estimator, torch.ones(1,3).cuda()]).requires_grad_(True)
        # n_linear = optimize_new_mat(_linear, lr=1e-3, L=3, nl=1, max_itr=10000)
        # scale = torch.tensor([[1],[1.2],[1],[0.5],[-.5]]).cuda()
        # optimize_new_mat(_linear, lr=1e-2, L=1, max_itr=10000)
        # new_mat = optimize_new_mat(torch.rand(5,3).cuda().requires_grad_(True), lr=1e-4, L=2, max_itr=10000)
        # new_mat = optimize_new_mat(new_linear, lr=1e-4, L=2, max_itr=10000)
        # new_mat = optimize_new_mat(new_mat, lr=1e-4, L=2)
        # new_mat = optimize_new_mat(new_mat, lr=1e-5)
        import pdb;pdb.set_trace()
        # new_linear = torch.tensor([[ 0.2960,  0.2050,  0.2060],[ 0.1850,  0.2840,  0.1710],[-0.1560,  0.1910,  0.2660],[-0.1820, -0.2690, -0.4710],[ 0.1753,  0.7383,  0.6385]], device='cuda:0', requires_grad=True)
        # svae_image(torch.einsum('bj,bik->jik', new_mat, latents_ref_img))
        # svae_image(torch.einsum('bj,bik->jik', new_linear, latents_ref_img))
        # svae_image(torch.einsum('bj,bik->jik', new_mat*.6+_linear*0.2, latents_ref_img))
        return
    
    def train_step(self, 
                   inputs,
                   input_masks,
                   ref_img,
                   ref_img_tensor,
                   guidance_scale=100,
                   use_clip = False,
                   clip_model= None,
                   ):
        """
        Args
            inputs (torch.Tensor):          [N, 4, 64, 64], rendered latent
            input_masks (torch.Tensor):     [N, 1, 64, 64], rendered latent mask
            ref_img (PIL.Image):            reference image path
            ref_img_tensor (torch.Tensor):  [N, 1, 512, 512],  reference image
        """
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        masks         = 1 - input_masks
        masked_images = inputs * masks        
        latents_grey_masks    = torch.cat([input_masks] * 4, dim=1) * self.latent_grey
        latents_masked_images = masked_images + latents_grey_masks
        
        
        # #### test!!!
        # self.approx_latent2rgb(ref_img)
            
        # # transforms.ToPILImage()(latents_ref_img[0]).save('tt.png')
        # vis = torch.cat([inputs[0], latents_masked_images[0], torch.cat([masks[0], masks[0], masks[0], masks[0]], dim=0)], dim=1)
        # # transforms.ToPILImage()(latents[0]).save('tt.png')
        # # transforms.ToPILImage()(latents_masked_images[0]).save('tt.png')
        # # transforms.ToPILImage()(masks[0]).save('tt.png')
        # transforms.ToPILImage()(vis).save('tt.png')
        # import pdb;pdb.set_trace()

        
        # --------------------------------------------------------------
        ref_img_clip = self.clip_image_process(ref_img)
        img_embeds, neg_img_embeds = self.image_encoder(ref_img_clip, return_uncond_vector=True)
        image_embeddings = torch.cat([neg_img_embeds, img_embeds])
        
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents) # (64 x 64 x 4)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            
            latent_model_input = torch.cat([latents_noisy] * 2)             # torch.Size([2, 4, 64, 64])
            latents_masked_images = torch.cat([latents_masked_images] * 2)  # torch.Size([2, 4, 64, 64])
            latents_masks = torch.cat([masks] * 2)                    # torch.Size([2, 1, 64, 64])
            # import pdb;pdb.set_trace()
            latent_model_input = torch.cat([latent_model_input,
                                            latents_masked_images,
                                            latents_masks], dim=1)
            # torch.Size([2, 9, 64, 64])
                        
            ### UNet2DModel,
            # noise_pred = self.unet(latent_model_input, t).sample
            ### UNet2DConditionModel, 
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=image_embeddings)['sample']
        
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # if use_clip:
        if use_clip and (t / self.num_train_timesteps) <= 0.4:
            self.scheduler.set_timesteps(self.num_train_timesteps)
            de_latents = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']
            imgs = self.denorm_img(self.decode_latents(de_latents))
            # import pdb;pdb.set_trace()
            loss = 10 * self.img_clip_loss(clip_model, imgs, ref_img_tensor)
            return loss
                    
        else:        
            # # clip grad for stable training?       
            # w(t), alpha_t * sigma_t^2
            w = (1 - self.alphas[t])
            # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
                
            # import pdb;pdb.set_trace()
            # manually backward, since we omitted an item in grad and cannot simply autodiff.
            # _t = time.time()
            latents.backward(gradient=grad, retain_graph=True)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
    
            return 0 # dummy 
        
    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()

    @staticmethod
    def normalize_img(tensor):
        return 2 * tensor - 1

    @staticmethod
    def denorm_img(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    @staticmethod
    def binarize_mask(masks):
        masks = (masks > 0.5).to(dtype=torch.float32)
        return masks



from transformers import CLIPPreTrainedModel, CLIPVisionModel

from diffusers.models.attention import BasicTransformerBlock

class PaintByExampleImageEncoder(CLIPPreTrainedModel):
    def __init__(self, config, proj_size=768):
        super().__init__(config)
        self.proj_size = proj_size

        self.model = CLIPVisionModel(config)
        self.mapper = PaintByExampleMapper(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, self.proj_size)

        # uncondition for scaling
        self.uncond_vector = nn.Parameter(torch.randn((1, 1, self.proj_size)))

    def forward(self, pixel_values, return_uncond_vector=False):
        clip_output = self.model(pixel_values=pixel_values)
        latent_states = clip_output.pooler_output
        latent_states = self.mapper(latent_states[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        if return_uncond_vector:
            return latent_states, self.uncond_vector

        return latent_states


class PaintByExampleMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(hid_size,
                                    num_heads,
                                    hid_size,
                                    activation_fn="gelu",
                                    attention_bias=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


# run main
if __name__ == "__main__":
    from PIL import Image
    from utils import seed_everything
    
    seed_everything(1234)

    import torchvision
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((512, 512))
    ])

    device = "cuda:0"

    guidance_scale = 7.5
    num_inference_steps = 25

    # load model
    # from src.paint_by_example import PaintbyExample
    model = PaintbyExample(device=device)

    ball_file = "/source/kseo/diffusion_playground/latent-nerf-test/data/monster_ball.jpg"
    # read iamge 
    img_file  = "/source/kseo/diffusion_playground/latent-nerf-test/img/image.png"
    mask_file = "/source/kseo/diffusion_playground/latent-nerf-test/img/mask.png"
    expl_file = "/source/kseo/diffusion_playground/latent-nerf-test/img/example.jpg"
    pil_img  = Image.open(img_file).convert("RGB").resize((512, 512))
    pil_mask = Image.open(mask_file).convert("RGB").resize((512, 512))
    
    # example_image = Image.open(expl_file).resize((512, 512))
    example_image = Image.open(ball_file).resize((512, 512))
    
    if False:
        # make mask
        import numpy as np
        pil_mask = np.zeros([512,512,3])
        pil_mask[200:400,100:400,:] = 255
        pil_mask = Image.fromarray(pil_mask.astype(np.uint8))


    # make it as tensor
    tensor_input_img   = transform(pil_img).unsqueeze(0).to(device)
    tensor_mask        = transform(pil_mask).unsqueeze(0).to(device)[:,:1,:,:]
    # tensor_example_img = transform(example_image).unsqueeze(0).to(device)

    # preprocess for clip
    tensor_example_img = model.clip_image_process(example_image)

    # generate
    with torch.no_grad():
        out = model(tensor_input_img, tensor_mask, tensor_example_img, num_inference_steps, guidance_scale)

    transforms.ToPILImage()(out[0]).save('tt.png')
    # show image
    # import mediapy as mp
    # mp.show_images({
    #     "input": pil_img,
    #     "mask": pil_mask,
    #     "example": example_image,
    # }, height=100)
    # mp.show_image(out[0].cpu().detach().numpy().transpose(1,2,0), height=250)
