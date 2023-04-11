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

MODEL_NAME = 'Fantasy-Studio/Paint-by-Example'
CACHE_DIR = "/source/skg/diffusion-project/hugging_cache"


class PaintbyExample(nn.Module):
    def __init__(self, device, \
        model_name=MODEL_NAME,\
        latent_mode=True,\
        cache_dir=CACHE_DIR):
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
        # self.image_encoder = CLIPVisionModel.from_pretrained(model_name, subfolder="image_encoder", cache_dir=cache_dir).to(self.device)
        self.image_encoder = PaintByExampleImageEncoder.from_pretrained(model_name, subfolder="image_encoder", cache_dir=cache_dir).to(self.device)
        
        self.model_id = "openai/clip-vit-base-patch32"
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
        self.scheduler = PNDMScheduler.from_pretrained(model_name, subfolder="scheduler", cache_dir=cache_dir, use_auth_token=self.token)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
       
        logger.info(f'\t successfully loaded stable diffusion!')
        self.linear_rgb_estimator = torch.tensor([
            #   R       G       B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ]).to(self.device)
        
        self.latent_grey = torch.tensor(
            #    L1      L2       L3      L4
            [[0.9071, -0.7711,  0.7437,  0.1510]]
        ).unsqueeze(-1).unsqueeze(-1).to(self.device)

    def clip_image_process(self, np_img):
        return self.feature_extractor(np_img, return_tensors="pt").pixel_values.to(self.device)

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

    def produce_latents(self,
            latents_masked_images, # [2, 4, 64, 64]
            latents_masks, # [2, 4, 64, 64]
            latents, # [1, 4, 64, 64]
            image_embeddings,
            guidance_scale=7.5):

        from tqdm import tqdm
        print("denoise...")
        with torch.autocast('cuda'):
            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # latents = [1, 4, 64, 64]
                
                latent_model_input = torch.cat([latents] * 2) 
                # -> extend in batch [2, 4, 64, 64]
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 
                # -> [2, 4, 64, 64]
                
                latent_model_input = torch.cat([latent_model_input,
                                                latents_masked_images,
                                                latents_masks], dim=1)
                # -> [2, 9, 64, 64]
                # import pdb;pdb.set_trace()
                
                
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,
                                            t,
                                            encoder_hidden_states=image_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)['prev_sample']
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
                
                if i % 5==0 and 0:
                    vis = torch.cat([
                            latents[0],
                            latents_masked_images[0],
                            torch.cat([latents_masks[0], latents_masks[0], latents_masks[0], latents_masks[0]], dim=0),
                        ], dim=1)
                    torchvision.utils.save_image(vis, 'ttt{:03}.png'.format(i))
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
        images (torch.Tensor): [B, 3, H, W]
        masks (torch.Tensor):  [B, 1, H, W]
        example_images (torch.Tensor): [B, 3, 224, 244] --> preprocess with clip_image_process
        num_inference_steps (int): number of inference steps
        guidance_scale (float): guidance scale
        strength (float): strength of the original input images
        latents (torch.Tensor): [B, 4, H//8, W//8]

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
        
        # decode
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # denormalize imgs
        imgs = self.denorm_img(imgs)
        return imgs

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start
    
    def train_step(self, 
                   inputs,              # latent
                   input_masks,         # latent mask
                   example_images,      # exampler image
                   guidance_scale=100,
                   ):
        """
        inputs (torch.Tensor):          [N, 4, 64, 64], rendered latent image
        input_masks (torch.Tensor):     [N, 1, 64, 64], rendered latent image
        example_images (PIL.Image):     [N, 3, H, W],   examplar image
        """
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
            
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        masks      = 1 - input_masks
        grey_masks = torch.cat([input_masks] * 4, dim=1) * self.latent_grey
        latents_masked_images =  inputs * masks # + grey_masks
        ## check 'latents_masked_images' image
        
        # import pdb;pdb.set_trace()
        # import numpy as np
        # rs_example_images = example_images.resize((512,512))
        # np_example_images = np.array(rs_example_images).transpose(2,0,1)
        # th_example_images = torch.tensor(np_example_images)[None] / 255.
        # th_example_images = th_example_images.to(self.device)
        # latents_example_images = self.encode_imgs(th_example_images) # torch.Size([1, 4, 64, 64])
        # # transforms.ToPILImage()(latents_example_images[0]).save('tt.png')
        # vis = torch.cat([inputs[0], latents_masked_images[0], torch.cat([masks[0], masks[0], masks[0], masks[0]], dim=0)], dim=1)
        # # transforms.ToPILImage()(latents[0]).save('tt.png')
        # # transforms.ToPILImage()(latents_masked_images[0]).save('tt.png')
        # # transforms.ToPILImage()(masks[0]).save('tt.png')
        # transforms.ToPILImage()(vis).save('tt.png')
        # import pdb;pdb.set_trace()

        
        # --------------------------------------------------------------
        # image clip embedding
        # example_images.min() -1.7923
        # example_images.max() 2.1459 
        # example_images.mean() 1.0935 
        # example_images.std() 1.2889
        example_images = self.clip_image_process(example_images)
        img_embeds, neg_img_embeds = self.image_encoder(example_images, return_uncond_vector=True)
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
            
            latent_model_input = torch.cat([latent_model_input,
                                            latents_masked_images,
                                            latents_masks], dim=1)
            # torch.Size([2, 9, 64, 64])
                        
            ### UNet2DModel,
            # noise_pred = self.unet(latent_model_input, t).sample
            ### UNet2DConditionModel, 
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings)['sample']
            
        
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
            
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
