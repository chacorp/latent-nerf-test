import sys
from pathlib import Path
from typing import Any, Dict, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from src import utils
from src.latent_paint_mesh.configs.train_config import TrainConfig
from src.latent_paint_mesh.training.views_dataset import (
    ViewsDataset, 
    circle_poses,
)
from src.stable_diffusion import StableDiffusion
from src.paint_by_example import PaintbyExample
from src.utils import make_path, tensor2numpy

import clip

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path           = make_path(self.cfg.log.exp_dir)
        self.ckpt_path          = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path  = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.clip_model, self.clip_preprocess = self.init_clip()
        self.mesh_model         = self.init_mesh_model()
        self.diffusion          = self.init_diffusion()
        
        self.lsize = self.cfg.render.train_grid_size
        
        # self.latent_w = torch.tensor(
        #     # [ 1.9546,  1.3794,  0.0308, -1.0842]
        #     [ 0.7441,  0.5251,  0.0117, -0.4128] ## normalized
        # )[None,:,None,None].repeat(1,1,self.lsize,self.lsize).to(self.device)        
        # self.latent_b = torch.tensor(
        #     [-0.8521, -2.5287,  1.1493,  1.2685]
        #     # [-0.2688, -0.7976,  0.3625,  0.4001] ## normalized
        # )[None,:,None,None].repeat(1,1,self.lsize,self.lsize).to(self.device)
        # self.latent_bw_max = self.latent_w - self.latent_b
        
        self.transform          = self.get_transform()
        self.text_z, self.text_z_H = self.ref_text_embeddings()
        
        ### reference image
        self.ref_image, self.ref_image_tensor, self.ref_image_embeds, self.ref_sampled_texture = self.get_image()
        
        with torch.no_grad():
            ref_samp_tex_encode = self.diffusion.encode_imgs(self.ref_sampled_texture)
            # ref_samp_tex_encode = F.interpolate(ref_samp_tex_encode, (128, 128), mode='bilinear')
            # noise = torch.randn_like(ref_samp_tex_encode)
            # ref_samp_tex_encode = (ref_samp_tex_encode + noise) * 0.5
            
        ref_samp_tex_encode.requires_grad = True
        self.mesh_model.texture_img = nn.Parameter(ref_samp_tex_encode)
        self.mesh_model.texture_img_rgb_finetune = nn.Parameter(self.ref_sampled_texture)
        # self.mesh_model.texture_img = nn.Parameter(self.ref_sampled_texture)
        
        ### set time step for latent sampling
        self.diffusion.scheduler.set_timesteps(self.cfg.guide.num_inference_steps)
        
        
        self.optimizer          = self.init_optimizer()
        self.dataloaders        = self.init_dataloaders()
        
        ## Optimizer for displacement
        self.optimizer_disp     = self.init_optimizer_disp()
        
        ### depricated ####################################################################
        # if not self.cfg.optim.use_SD:
        #     ## get clip mean & std for normalization
        #     self.clip_mean      = torch.tensor(self.diffusion.feature_extractor.image_mean).unsqueeze(-1).unsqueeze(-1).to(self.device)
        #     self.clip_std       = torch.tensor(self.diffusion.feature_extractor.image_std).unsqueeze(-1).unsqueeze(-1).to(self.device)
        #     self.normalize_clip = lambda x, mu, std: (x - mu) / std
        ### depricated ####################################################################
        
        
        
        self.ref_pose           = self.get_reference_pose()
        self.criterionL1        = nn.L1Loss()
        self.criterionL2        = nn.MSELoss()
        self.criterionCLIP      = torch.nn.CosineSimilarity(dim=1, eps=1e-12)
        self.downSample         = nn.Upsample(scale_factor=0.125, mode='nearest')
        
        self.normalize_clip     = transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        
        self.past_checkpoints   = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        

    def init_mesh_model(self) -> nn.Module:
        if self.cfg.render.backbone == 'texture-mesh':
            from src.latent_paint_mesh.models.textured_mesh import TexturedMeshModel
            model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                      latent_mode=True, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)
        elif self.cfg.render.backbone == 'texture-rgb-mesh':
            from src.latent_paint_mesh.models.textured_mesh import TexturedMeshModel
            model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                      latent_mode=False, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)
        else:
            raise NotImplementedError(f'--backbone {self.cfg.render.backbone} is not implemented!')

        model = model.to(self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    # def init_diffusion(self) -> StableDiffusion:
    def init_diffusion(self):
        # text-guided 
        if self.cfg.optim.use_SD:
            MODEL_NAME = self.cfg.guide.diffusion_name
            CACHE_DIR = "/source/kseo/huggingface_cache"
            diffusion_model = StableDiffusion(
                    self.device, 
                    model_name  = MODEL_NAME,
                    cache_dir   = CACHE_DIR,
                    latent_mode = self.mesh_model.latent_mode,
                    min_step    = self.cfg.optim.min_step,
                    max_step    = self.cfg.optim.max_step,
                )
        else:
            MODEL_NAME = self.cfg.guide.paint_by_example
            CACHE_DIR = "/source/kseo/huggingface_cache"
            diffusion_model = PaintbyExample(
                    self.device, 
                    model_name  = MODEL_NAME,
                    cache_dir   = CACHE_DIR,
                    latent_mode = self.mesh_model.latent_mode,
                )
            
        for p in diffusion_model.parameters():
            p.requires_grad = False
            
        return diffusion_model

    def init_clip(self):
        # Load the model
        # model, preprocess = clip.load('ViT-B/32', self.device)
        if self.cfg.optim.use_SD:
            model, preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        else:
            model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        clip.model.convert_weights(model)
        return model, preprocess
        
    def _clip_image_embeddings(self, image):
        """
        Args:
            image (PIL.Image or torch.Tensor): input image 
        returns:
            image_features (torch.Tensor): 
        """
        # Prepare the inputs (convert PIL.Image to torch.Tensor)
        if type(image) == torch.Tensor:
            if image.shape[-1] != 224:
                image = F.interpolate(image, (224, 224), mode='bilinear')
            image_input = self.normalize_clip(image)
        else:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Calculate features
        # image_features = self.clip_model.encode_image(image_input)
        image_features = self.clip_model.encode_image(image_input).type(self.diffusion.unet.dtype)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def clip_image_embeddings(self, image, use_grad=False):
        if use_grad:
            return self._clip_image_embeddings(image)
        else:
            with torch.no_grad():
                return self._clip_image_embeddings(image)
        
    def ref_text_embeddings(self):
        ref_text = self.cfg.guide.text
        use_grad = self.cfg.optim.use_opt_txt
        
        ####
        # TODO:
        # works with transformer CLIP self.diffusion.get_text_embeds
        # does not work with openAI clip -> why?
            
        if not self.cfg.guide.append_direction:
            # text_z = self.diffusion.get_text_embeds([ref_text]) # torch.Size([2, 77, W]) 0: uncond 1: cond
            text_z = self.get_text_embeddings([ref_text], use_grad=use_grad)
            return text_z
        else:
            text_z = []
            text_z_head = []
            for d in ['front', 'left side', 'back', 'right side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_head = f"a close up face of {ref_text}, {d} view"
                # text_z.append(self.diffusion.get_text_embeds([text]))
                text_z.append(self.get_text_embeddings([text], use_grad=use_grad))
                text_z_head.append(self.get_text_embeddings([text_head], use_grad=use_grad))
            text_z = torch.stack(text_z)
            text_z_head = torch.stack(text_z_head)
            
            return text_z, text_z_head
    
    # grad flow!
    def encode_text_embedding(self, tkn):
        """
        Args:
            tkn (torch.tensor): [B, 77] text token
        Return:
            x (torch.tensor):   [B, 77, W] text embeddings [batch_size, n_ctx, transformer.width: 512(ViT-B/32) / 768 (ViT-L/14)]
        """
        x = self.clip_model.token_embedding(tkn).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.diffusion.unet.dtype) # match type
        
        # x = x[torch.arange(x.shape[0]), tkn.argmax(dim=-1)] @ self.clip_model.text_projection
        # print(x.shape)
        return x
    
    def pooled_text_feature(self, text_z, tkn):
        """
        Args:
            text_z (torch.tensor): [B, 77, W] text feature [batch_size, n_ctx, transformer.width:(ViT-B/32) / 768 (ViT-L/14)]
            tkn (torch.tensor): [B, 77] text token
        Return:
            pooled feature; take features from the eot embedding (eot_token is the highest number in each sequence)
        """
        return text_z[torch.arange(text_z.shape[0]), tkn.argmax(dim=-1)] @ self.clip_model.text_projection.type(self.diffusion.unet.dtype)
    
    def _get_text_embeddings(self, text):
        """
        Args:
            text (list[str]): text prompt
        Return:
            text_embeddings (torch.tensor): [2, 77, W] text feature
        """
        text_ = clip.tokenize(text).to(self.device)
        text_embedding = self.encode_text_embedding(text_)
        
        uncond = clip.tokenize([''] * len(text)).to(self.device)
        uncond_embedding = self.encode_text_embedding(uncond)
        
        text_embeddings = torch.cat([uncond_embedding, text_embedding])
        return text_embeddings

    def get_text_embeddings(self, text, use_grad=False):
        if use_grad:
            return self._get_text_embeddings(text)
        else:
            with torch.no_grad():
                return self._get_text_embeddings(text)        

    def get_image(self) -> torch.Tensor:
        image = Image.open(self.cfg.guide.image).convert('RGB')
        image_tensor = self.transform(image)[None].to(self.device)
        image_embeds = self.clip_image_embeddings(image)
        
        image_sampled = Image.open(self.cfg.guide.sampled_texture).convert('RGB')
        image_sampled = image_sampled.resize((512,512), resample=Image.Resampling.LANCZOS)        
        image_sampled = transforms.ToTensor()(image_sampled)[None].to(self.device)
        
        return image, image_tensor, image_embeds, image_sampled
    
    def get_transform(self):
        return  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    
    # Hard coded for experiment
    def get_reference_pose(self):
        ## sphere
        # radius = 1.60
        ## SMPL
        radius = 1.2
        thetas = 60.0
        phis   = -20.0 # minus left plus right
        dirs, thetas, phis, radius = circle_poses(self.device, radius=radius, theta=thetas, phi=phis)
        data = {
            'dir'    : dirs,
            'theta'  : thetas,
            'phi'    : phis,
            'radius' : radius,
        }
        return data
    
    def init_optimizer(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)
        return optimizer
    
    def init_optimizer_disp(self) -> Optimizer:
        return torch.optim.Adam([self.mesh_model.displacement], lr=self.cfg.optim.disp_lr, betas=(0.9, 0.99), eps=1e-15)
        # return torch.optim.Adam([self.mesh_model.xi], lr=self.cfg.optim.disp_lr, betas=(0.9, 0.99), eps=1e-15)
        # return torch.optim.Adam([*self.mesh_model.MLP.parameters()], lr=self.cfg.optim.disp_lr, betas=(0.9, 0.99), eps=1e-15)

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader   = ViewsDataset(self.cfg.render, device=self.device, type='train', size=100).dataloader()
        val_loader         = ViewsDataset(self.cfg.render, device=self.device, type='val', size=self.cfg.log.eval_size).dataloader()
        
        # Will be used for creating the final video
        val_large_loader   = ViewsDataset(self.cfg.render, device=self.device, type='val', size=self.cfg.log.full_eval_size).dataloader()
        
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        # dataloaders_H = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    
    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        step_weight = 0
        prev_optim_step = 0
        while self.train_step < self.cfg.optim.iters:
            
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                                
                self.train_step += 1
                pbar.update(1)

                # prev_disp = self.mesh_model.displacement.clone()
                self.optimizer.zero_grad()
                self.optimizer_disp.zero_grad()
                descrition = ""
                
                # data['image_z'] = self.image_z
                # pred_rgbs, loss = self.train_render(data)
                
                # CLIPD = self.train_step % 10 == 1
                # CLIPD = np.random.uniform(0, 1) < 0.5
                CLIPD = False
                # CLIPD = True
                
                use_clip = True
                # log = np.random.uniform(0, 1) < 0.05
                log = self.train_step % 50 == 0
                
                # if False: #self.train_step < 100:
                # if self.train_step < 100:
                if False:
                    ### Image Reconstruction loss
                    descrition += "type 1: "
                    reconstruction = True
                    # use_clip = False
                    pred_rgbs, lap_loss, loss_guidance     = self.train_render_clip(data, use_clip=use_clip, log=log)
                else:
                    # if CLIPD:
                    if False:
                        # code reference: https://github.com/junshutang/Make-It-3D
                        ### SDS + CLIP-D 
                        descrition += "type 2: "
                        pred_rgbs, lap_loss, loss_guidance = self.train_render_clip(data, use_clip=use_clip, log=log)
                    else:
                        ### SDS only
                        descrition += "type 3: "
                        pred_rgbs, lap_loss, loss_guidance = self.train_render_text(data)
                self.optimizer.step()

                
                
                # # lap_loss = (lap_loss * laplacian_weight)
                # # ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L30
                # # lap_loss =  10. ** 4 * lap_loss #/ (1 + self.train_step)
                # lap_loss = (self.cfg.optim.lap_weight ** 2) * lap_loss / (1 + self.train_step)
                # # 'lap': lambda cst, it: 2000**2*cst / (1 + it)

                ## Offset regularization
                ## ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L31
                
                # displacement = self.mesh_model.Linv.mm(self.mesh_model.xi)
                # displacement = 0 # self.mesh_model.MLP(self.mesh_model.init_lap)
                # displacement = self.mesh_model.displacement
                # reg_loss = torch.mean(torch.mean(displacement[None]**2, axis=1), axis=1)
                # reg_loss = (10. ** self.cfg.optim.reg_weight) * reg_loss # / (1 + (step_weight))
                
                # 'offsets' = torch.mean(torch.mean(smpl.offsets ** 2, axis=1))
                
                ### color loss
                # pred_tex = self.diffusion.decode_latents(self.mesh_model.texture_img)
                # color_loss = self.criterionL1(self.mesh_model.texture_img_rgb_finetune, pred_tex)
                
                # # loss = lap_loss + reg_loss + loss_guidance
                # loss = lap_loss
                # loss = color_loss
                # loss.backward()
                # ## optimize displacement interval
                # self.optimizer_disp.step()
                
                # new_disp = torch.mean(displacement)
                # descrition += "disp:{:05f} ".format(new_disp)                
                # descrition += "lap: {:05f} reg:{:05f} ".format(lap_loss.item(), reg_loss.item())
                descrition += "lap: {:05f}".format(lap_loss.item())
                if use_clip and CLIPD:
                    descrition += "clip:{:05f}".format(loss_guidance.item())
                    
                # ========= use optim_step =========
                # descrition += "disp:{:06f} disp_step:{}".format(new_disp, next_optim_step)
                
                pbar.set_description(descrition)

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.mesh_model.train()
                    step_weight +=1

                # if np.random.uniform(0, 1) < 0.05:
                # if log and not CLIPD:
                if log:
                # if False:
                    # not pixelwise loss :: 'pred_rgbs' should be latent (4 channel)
                    # Randomly log rendered images throughout the training
                    # logger.info('view: theta={}, phi={}, radius={}'.format(data['theta'], data['phi'], data['radius']))
                    self.log_train_renders(pred_rgbs)
                    
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)
        
        log_idx = self.train_step // self.cfg.log.save_interval
        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                # Image.fromarray(pred).save(save_path / f"step_{self.train_step:05d}_{i:04d}_rgb.png")
                Image.fromarray(pred).save(save_path / f"step_{log_idx:05d}_{i:04d}_rgb.png")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        
        # Image.fromarray(texture).save(save_path / f"step_{self.train_step:05d}_texture.png")
        Image.fromarray(texture).save(save_path / f"step_{log_idx:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.train_step:05d}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self):
        try:
            self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)
        except:
            logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path, guidance=self.diffusion)

            logger.info(f"\tDone!")

    def train_render_clip(self, data, use_clip=False, log=False):
        if use_clip:
            theta       = data['theta']
            phi         = data['phi']
            radius      = data['radius']
        else:
            theta       = self.ref_pose['theta']
            phi         = self.ref_pose['phi']
            radius      = self.ref_pose['radius']
        
        ## render latent
        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        
        pred_rgb        = outputs['image']      # [B, 3, H, W]
        pred_rgb_mask   = outputs['mask']       # [B, 1, H, W]
        lap_loss        = outputs['lap_loss']
        
        
        img = torch.einsum('bi,abcd->aicd', self.diffusion.linear_rgb_estimator, pred_rgb) * 0.5 + 0.5
        ref = self.downSample(self.ref_image_tensor)
        loss_guidance = self.diffusion.img_clip_loss(self.clip_model, ref, img) ## cos-sim
        
        
        loss_guidance.backward(retain_graph=True)
        
        return pred_rgb, lap_loss, loss_guidance

    
    # Hard coded for experiment
    def get_camera_pose(self, 
                        radius = 1.2, 
                        thetas = 60.0, 
                        phis   = -20, # minus left plus right
                       ):
        ## sphere
        # radius = 1.60
        ## SMPL
        dirs, thetas, phis, radius = circle_poses(self.device, radius=radius, theta=thetas, phi=phis)
        data = {
            'dir'    : dirs,
            'theta'  : thetas,
            'phi'    : phis,
            'radius' : radius,
        }
        return data
    
    def train_render_text(self, data: Dict[str, Any]):
        thetas    = data['theta']
        phis      = data['phi']
        rs   = data['radius']
        
        # theta       = self.ref_pose['theta']
        # phi         = self.ref_pose['phi']
        # radius      = self.ref_pose['radius']
        
        ## randomly select view: body, head
        is_body = np.random.uniform(0, 1) < 0.5
    
        ## render
        # outputs  = self.mesh_model.render(thetas, phis, rs, dims=(64, 64), is_body=is_body)
        outputs  = self.mesh_model.render(thetas, phis, rs, dims=(self.lsize, self.lsize), is_body=is_body)
        outputs_RGB= self.mesh_model.render_RGB(thetas, phis, rs, self.mesh_model.texture_img_rgb_finetune, dims=(512, 512), is_body=is_body)
        
        ## rendered w/ current texture
        pred_rgb        = outputs['image']      # [B, 4, H, W]
        pred_rgb_mask   = outputs['mask']       # [B, 1, H, W]
        pred_lighting   = outputs['lighting']   # [B, 1, H, W]
        pred_normal     = outputs['normal']   # [B, 3, H, W]
        lap_loss        = outputs['lap_loss']
        # pred_back = outputs['background']
        
        
        # Guidance loss
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] if is_body else self.text_z_H[dirs]
        else:
            text_z = self.text_z
            
        
#         latent_shading = pred_lighting*self.latent_bw_max + self.latent_b        
#         latent = pred_rgb*pred_lighting + latent_shading        
    
        # pred_rgb = torch.einsum('ib,abcd->aicd', self.diffusion.linear_rgb_estimator, pred_rgb) ## becomes noisy
        
        
        ## TADA! Text to Animatable Digital Avatars
        # alpha = 0.5
        # latent_I = self.diffusion.encode_imgs(pred_rgb*pred_lighting)
        # self.mesh_model.texture_img
                
        # latent_I_ = latent_I.clone().detach()
        # latent_I_.requires_grad = True
        # latent_N = self.diffusion.encode_imgs(pred_normal) * alpha + latent_I_ * (1 - alpha)
        """
            TODO:
            - [v] DDIM inversion
            - [v] null-text inversion : get optimal 'text_z' (uncond_embeddings_list) - carefull with the guidance_scale
            - [v] better initialization for texture ...? 
                - invert image and sample image noise to texture space
                - just encode sampled_texture from SamplerNet
                
            - [ ] add Jacobain optimization (ref:textdeformer)...?
            - [ ] denoise with geometry constraint (follow 'pred_rgb')
                - RGB/normal -> [Enc] -> latent -> [diffuse] -> SDS : does not work... (ref: TADA!)
                - and lighting
                
            - [v] import Dreambooth checkpoint
            - [v] SDedit .... why good quality?
                - > use it for delta denosing score?
            + upgrade diffusers > 0.17.0
            + how about image loss?
        """
        ## even: uncond, odd:cond
        # torch.stack(torch.arange(12).chunk(6))
        # tensor([[ 0,  1],
        #         [ 2,  3],
        #         [ 4,  5],
        #         [ 6,  7],
        #         [ 8,  9],
        #         [10, 11]])    
        # torch.stack(torch.arange(12).chunk(6)).permute(1,0).reshape(-1)
        # tensor([ 0,  2,  4,  6,  8, 10,  1,  3,  5,  7,  9, 11])
        # loss_guidance = self.diffusion.train_step(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), pred_rgb)
        # import pdb;pdb.set_trace()
        # noise = torch.randn_like(pred_rgb)
        
        
        #### SDedit ....??? why good result?
        # latent_RGB = self.diffusion.encode_imgs(outputs_RGB['image'])
        # latent_RGB = self.diffusion.encode_imgs(outputs_RGB['image']*outputs_RGB['lighting'])
        
        # import pdb;pdb.set_trace()
        # start  = 40
        # t_step = self.diffusion.scheduler.timesteps[start]
        # noise  = torch.randn_like(pred_rgb)
        
        # latent_RGB_n = self.diffusion.scheduler.add_noise(latent_RGB, noise, t_step)
        # latent_RGB_n_ = self.diffusion.produce_latents(text_z[0], latents=latent_RGB_n, start=start)
        # Image.fromarray(tensor2numpy(self.diffusion.decode_latents(latent_RGB_n_).permute(0, 2, 3, 1)[0])).save('t.png')
        
        
        # latent_I_n = self.diffusion.scheduler.add_noise(pred_rgb, noise, t_step)
        # latent_I_n_ = self.diffusion.produce_latents(text_z[0], latents=latent_I_n, start=start)
        # Image.fromarray(tensor2numpy(self.diffusion.decode_latents(latent_I_n_).permute(0, 2, 3, 1)[0])).save('t.png')
        
        # Image.fromarray(tensor2numpy(self.diffusion.decode_latents(pred_rgb).permute(0, 2, 3, 1)[0])).save('t.png')
        # Image.fromarray(self.diffusion.embeds_to_img(text_z[0], latents=latent_RGB_n_, start=start)[0]).save('t.png')
                
        # Image.fromarray(tensor2numpy(outputs_RGB['image'].permute(0,2,3,1)[0])).save('t.png')
        # Image.fromarray(tensor2numpy(pred_rgb.permute(0,2,3,1)[0])).save('t.png')
        
        # Image.fromarray(tensor2numpy(self.diffusion.decode_latents(latent_I).permute(0, 2, 3, 1)[0])).save('t.png')
        
        # latents = self.diffusion.produce_latents(text_z[0], latents=pred_rgb, start=40)
        # Image.fromarray(tensor2numpy(self.diffusion.decode_latents(latents).permute(0, 2, 3, 1)[0])).save('t.png')

        I_grad = self.diffusion.train_step_delta(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), outputs['image'], outputs_RGB['image'])
        # I_grad = self.diffusion.train_step(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), latent_I)
        
        # N_grad = self.diffusion.train_step(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), latent_N)
        # https://github.com/threestudio-project/threestudio/blob/8a51c37317b6f7cd74bb3cb24c975b56d0a96703/threestudio/models/guidance/stable_diffusion_guidance.py#L427C9-L427C18
        # tex_loss = self.criterionL2((latent_I-I_grad), I_grad)
        # nrm_loss = self.criterionL2((latent_N-N_grad), N_grad)
        
#         total_loss = tex_loss + nrm_loss
#         total_loss.backward()
        # I_grad_up = F.interpolate(I_grad, size=pred_rgb.shape[-1], mode='bilinear', align_corners=True)
        # N_grad_up = F.interpolate(N_grad, size=pred_rgb.shape[-1], mode='bilinear', align_corners=True)
        # pred_rgb.backward(gradient=I_grad_up, retain_graph=True)
        # pred_normal.backward(gradient=N_grad_up)
        
        outputs['image'].backward(gradient=I_grad)
        
        # latent_N.backward(gradient=N_grad, retain_graph=True)
        # return pred_rgb, lap_loss, loss_guidance
        
        # I_grad = self.diffusion.train_step(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), pred_rgb)
        # I_grad = self.diffusion.train_step(text_z[0], pred_rgb, guidance_scale=7.5)
        # I_grad = self.diffusion.train_step(text_z.permute(1, 0, 2, 3).reshape(-1, 77, 768), latent, guidance_scale=7.5)
        # pred_rgb.backward(gradient=I_grad*pred_rgb_mask)
        
        return pred_rgb, lap_loss, torch.zeros([1]).to(self.device)

    def eval_render(self, data):
        theta    = data['theta']
        phi      = data['phi']
        radius   = data['radius']
        # import pdb;pdb.set_trace()
        dim      = self.cfg.render.eval_grid_size
        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents, test=True, dims=(dim,dim))
        # outputs  = self.mesh_model.render_RGB(theta, phi, radius, self.mesh_model.texture_img, dims=(dim,dim))
        
        pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        pred_rgb_mask = outputs['mask'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return pred_rgb, texture_rgb

    def log_train_renders(self, preds: torch.Tensor):
        if self.mesh_model.latent_mode: ## rendered image = latent
        # if False: ## rendered image = RGB
            pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
            if not self.cfg.optim.use_SD:
                pred_rgb = self.diffusion.denorm_img(pred_rgb)
        else:
            pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.png'
        # save_path = self.train_renders_path / f'step_{log_idx:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred_rgb = tensor2numpy(pred_rgb[0])

        Image.fromarray(pred_rgb).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        def decode_texture_img(latent_texture_img):
            decoded_texture = self.diffusion.decode_latents(latent_texture_img)
            decoded_texture = F.interpolate(decoded_texture,
                                            (self.cfg.guide.texture_resolution, self.cfg.guide.texture_resolution),
                                            mode='bilinear', align_corners=False)
            return decoded_texture

        if 'model' not in checkpoint_dict:
            if not self.mesh_model.latent_mode:
                # initialize the texture rgb image from the latent texture image
                checkpoint_dict['texture_img_rgb_finetune'] = decode_texture_img(checkpoint_dict['texture_img'])
            self.mesh_model.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        if not self.mesh_model.latent_mode:
            # initialize the texture rgb image from the latent texture image
            checkpoint_dict['model']['texture_img_rgb_finetune'] = \
            decode_texture_img(checkpoint_dict['model']['texture_img'])

        missing_keys, unexpected_keys = self.mesh_model.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()

        state['model'] = self.mesh_model.state_dict()

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)
