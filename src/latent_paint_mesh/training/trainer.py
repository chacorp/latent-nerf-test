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
from src.latent_paint_mesh.training.views_dataset import ViewsDataset, circle_poses
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
        
        ### depricated ####################################################################
        if not self.cfg.optim.use_SD:
            ## get clip mean & std for normalization
            self.clip_mean      = torch.tensor(self.diffusion.feature_extractor.image_mean).unsqueeze(-1).unsqueeze(-1).to(self.device)
            self.clip_std       = torch.tensor(self.diffusion.feature_extractor.image_std).unsqueeze(-1).unsqueeze(-1).to(self.device)
            self.normalize_clip = lambda x, mu, std: (x - mu) / std
        ### depricated ####################################################################
        
        # self.text_z             = self.calc_text_embeddings()
        self.text_z             = self.ref_text_embeddings()
        # self.image_z            = self.calc_image_embeddings()
        # self.image_z, self.image= self.calc_image_embeddings()
        self.image_z            = self.ref_image_embeddings()
        
        self.optimizer          = self.init_optimizer()
        self.dataloaders        = self.init_dataloaders()
        self.transform          = self.get_transform()
        
        ### reference image
        self.ref_image, self.ref_image_tensor = self.get_image()
        self.ref_image_embeds = self.clip_image_embeddings(self.ref_image)
        # if self.cfg.optim.use_SD:
        #     # self.ref_image_embeds, _ = self.calc_image_embeddings()
        #     self.ref_image_embeds = self.clip_image_embeddings(self.ref_image)
        # else:
        #     self.ref_image_embeds   = self.get_image_embedding()
        self.ref_pose           = self.get_reference_pose()
        self.criterionL1        = nn.L1Loss()
        self.criterionL2        = nn.MSELoss()
        self.criterionCLIP      = torch.nn.CosineSimilarity(dim=1, eps=1e-12)
        
        self.clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.normalize_clip = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
        ## Optimizer for displacement
        self.optimizer_disp     = self.init_optimizer_disp()
        self.past_checkpoints   = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        

    def init_mesh_model(self) -> nn.Module:
        if self.cfg.render.backbone == 'texture-mesh':
            # from src.latent_paint.models.textured_mesh import TexturedMeshModel
            from src.latent_paint_mesh.models.textured_mesh import TexturedMeshModel
            model = TexturedMeshModel(self.cfg, device=self.device, render_grid_size=self.cfg.render.train_grid_size,
                                      latent_mode=True, texture_resolution=self.cfg.guide.texture_resolution).to(self.device)
        elif self.cfg.render.backbone == 'texture-rgb-mesh':
            # from src.latent_paint.models.textured_mesh import TexturedMeshModel
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
            MODEL_NAME = '/source/kseo/huggingface_cache/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a'
            CACHE_DIR = "/source/kseo/huggingface_cache"
            diffusion_model = StableDiffusion(
                    self.device, 
                    # model_name   = self.cfg.guide.diffusion_name,
                    # concept_name = self.cfg.guide.concept_name,
                    model_name   = MODEL_NAME,
                    # cache_dir    = CACHE_DIR,
                    latent_mode  = self.mesh_model.latent_mode,
                )
        else:
            MODEL_NAME = 'Fantasy-Studio/Paint-by-Example'
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
            image = F.interpolate(image, (224, 224), mode='bilinear')
            image_input = self.normalize_clip(image)
        else:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Calculate features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        
        # Normalize features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def clip_image_embeddings(self, image, use_grad=False):
        if use_grad:
            return self._clip_image_embeddings(image)
        else:
            with torch.no_grad():
                return self._clip_image_embeddings(image)
        
    def ref_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        use_grad = self.cfg.optim.use_opt_txt
        
        ####
        # TODO:
        # works with transformer CLIP self.diffusion.get_text_embeds
        # does not work with openAI clip -> why?
            
        if not self.cfg.guide.append_direction:
            # text_z = self.diffusion.get_text_embeds([ref_text]) # torch.Size([2, 77, W]) 0: uncond 1: cond
            text_z = self.get_text_embeddings([ref_text], use_grad=use_grad)
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                # text_z.append(self.diffusion.get_text_embeds([text]))
                text_z.append(self.get_text_embeddings([text], use_grad=use_grad))
            # import pdb;pdb.set_trace()
        return text_z

    def ref_image_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.clip_image_embeddings(Image.open(self.cfg.guide.image))
        # ref_image = self.cfg.guide.image
        # import pdb;pdb.set_trace()
        # image_z, image = self.diffusion.get_image_embeds(ref_image)
        # return image_z, image[None]
        
    ### depricated ######################################################################################
    # reference image embeding for CLIP loss (use it from paint-by-example)
    def get_image_embedding(self):
        with torch.no_grad():
            ref_img = self.diffusion.feature_extractor(images=self.ref_image, return_tensors="pt").pixel_values
            ref_image_embeds = self.diffusion.image_encoder(ref_img.to(self.device))
            ref_image_embeds = ref_image_embeds.squeeze(1)
            ref_image_embeds = ref_image_embeds / ref_image_embeds.norm(p=2, dim=-1, keepdim=True)
        return ref_image_embeds.detach()
    ### depricated ######################################################################################
    
    
    # grad flow!
    def encode_text_embedding(self, tkn):
        """
        Args:
            tkn (torch.tensor): [B, 77] text token
        Return:
            x (torch.tensor): [B, 77, W] text embeddings [batch_size, n_ctx, transformer.width:(ViT-B/32) / 768 (ViT-L/14)]
        """        
        x = self.clip_model.token_embedding(tkn).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = self.clip_model.ln_final(x).type(self.diffusion.unet.dtype) # match type
        # print(x.shape)
        return x
    
    def _get_text_embeddings(self, text):
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
        
    def pooled_text_feature(self, ftr, tkn):
        """
        Args:
            ftr (torch.tensor): [B, 77, W] text feature [batch_size, n_ctx, transformer.width:(ViT-B/32) / 768 (ViT-L/14)]
            tkn (torch.tensor): [B, 77] text token
        Return:
            pooled feature; take features from the eot embedding (eot_token is the highest number in each sequence)
        """
        return ftr[torch.arange(ftr.shape[0]), tkn.argmax(dim=-1)] @ self.clip_model.text_projection
    

    def get_image(self) -> torch.Tensor:
        image = Image.open(self.cfg.guide.image)
        # return self.transform(image)
        return image, self.transform(image).to(self.device)
    
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
        radius = 1.0
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

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = ViewsDataset(self.cfg.render, device=self.device, type='train', size=100).dataloader()
        val_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device, type='val',
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
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

        ### TODO:
        ## 1. optimize text_x that best represents the image using CLIP CosSim loss
        # image_text_z = self.diffusion.optimize_text_token([self.cfg.guide.text], self.image_z, self.image, max_itr=100)
        # if self.cfg.optim.use_opt_txt:
        #     self.text_z = self.diffusion.optimize_text_token(self.text_z, self.image_z, max_itr=100)
        # import pdb;pdb.set_trace()
        ## 2. use Paint-by-Example

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        step_weight = 0
        prev_optim_step = 0
        while self.train_step < self.cfg.optim.iters:
            
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                
                ## Laplace loss weighting
                ## ref: https://github.com/NasirKhalid24/CLIP-Mesh/blob/d3cf57ebe5e619b48e34d6f0521a31b2707ddd72/configs/paper.yml
                # if self.train_step == 0:
                #     laplacian_weight = self.cfg.optim.laplacian_weight
                #     laplacian_min = self.cfg.optim.laplacian_min
                # else:
                #     laplacian_weight = (laplacian_weight - laplacian_min) * 10.**(-self.train_step*1e-06) + laplacian_min
                
                self.train_step += 1
                pbar.update(1)

                # prev_disp = self.mesh_model.displacement.clone()
                self.optimizer.zero_grad()
                self.optimizer_disp.zero_grad()
                descrition = ""
                
                # data['image_z'] = self.image_z
                # pred_rgbs, loss = self.train_render(data)
                
                # reconstruction = self.train_step % 10 == 1
                # reconstruction = self.train_step % 10 == 1
                # reconstruction = np.random.uniform(0, 1) < 0.5
                # reconstruction = True
                reconstruction = False
                
                use_clip = True
                log = np.random.uniform(0, 1) < 0.05
                
                # if False: #self.train_step < 100:
                # if self.train_step < 100:
                if False:
                    ### Image Reconstruction loss
                    descrition += "step 1: "
                    reconstruction = True
                    # use_clip = False
                    pred_rgbs, lap_loss, loss_guidance     = self.train_render_clip(data, use_clip=use_clip, log=log)
                else:
                    if reconstruction:
                        ### Image Reconstruction loss
                        descrition += "step 2: "
                        # use_clip = np.random.uniform(0, 1) < 0.5
                        pred_rgbs, lap_loss, loss_guidance = self.train_render_clip(data, use_clip=use_clip, log=log)
                    else:
                        ### Score Distillation Sampling
                        descrition += "step 3: "
                        pred_rgbs, lap_loss, loss_guidance = self.train_render_text(data)
                # sds_grad = self.mesh_model.displacement.grad
                
                self.optimizer.step()
                
                # import pdb;pdb.set_trace()
                # optim_step = int(50 * 10.**(-self.train_step*5e-05)+1)
                
                # if self.train_step % optim_step == 0:
                                
                # ========= use optim_step =========
                # next_optim_step = optim_step + prev_optim_step
                # if self.train_step >= next_optim_step:
                #     prev_optim_step = next_optim_step
                #     lap_loss =  10. ** 4 * lap_loss #/ (1 + step_weight)

                #     ## Offset regularization
                #     # reg_loss = self.mesh_model.displacement.norm(dim=1).mean()
                #     reg_loss = torch.mean(torch.mean(self.mesh_model.displacement[None]**2, axis=1), axis=1)
                #     reg_loss = 10. ** self.cfg.optim.reg_weight * reg_loss # / (1 + (step_weight))
                    
                #     loss = lap_loss + reg_loss
                #     loss.backward()
                #     ## optimize displacement interval
                #     self.optimizer_disp.step()
                
                
                # lap_loss = (lap_loss * laplacian_weight)
                # ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L30
                lap_loss =  10. ** 4 * lap_loss #/ (1 + step_weight)

                ## Offset regularization
                ## ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L31
                # reg_loss = self.mesh_model.displacement.norm(dim=1).mean()
                reg_loss = torch.mean(torch.mean(self.mesh_model.displacement[None]**2, axis=1), axis=1)
                reg_loss = 10. ** self.cfg.optim.reg_weight * reg_loss # / (1 + (step_weight))
                
                loss = lap_loss + reg_loss + loss_guidance
                loss.backward()
                ## optimize displacement interval
                self.optimizer_disp.step()
                
                new_disp = torch.mean(self.mesh_model.displacement)
                descrition += "disp:{:05f} ".format(new_disp)                
                descrition += "lap: {:05f} reg:{:05f} ".format(lap_loss.item(), reg_loss.item())
                if use_clip and reconstruction:
                    descrition += "clip:{:05f} ".format(loss_guidance.item())
                    
                # ========= use optim_step =========
                # descrition += "disp:{:06f} disp_step:{}".format(new_disp, next_optim_step)
                # if self.train_step % optim_step == 0:
                #     descrition += " lap: {:06f} reg:{:06f}".format(lap_loss.item(), reg_loss.item())
                
                pbar.set_description(descrition)

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.mesh_model.train()
                    step_weight +=1

                # if np.random.uniform(0, 1) < 0.05:
                if log and not reconstruction:
                    # not pixelwise loss :: 'pred_rgbs' should be latent (4 channel)
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs)
                    
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.train_step:05d}_{i:04d}_rgb.png")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.train_step:05d}_texture.png")

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
        dim             = self.cfg.render.eval_grid_size
        
        ## render latent
        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        ## render image
        # outputs2  = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
        #     decode_func = self.diffusion.decode_latents_grad,
        #     dims        = (dim, dim), 
        #     ref_view    = True)
        
        ## rendered w/ current texture
        pred_rgb        = outputs['image']      # [B, 3, H, W]
        pred_rgb_mask   = outputs['mask']       # [B, 1, H, W]
        lap_loss        = outputs['lap_loss']
        
        # pred_rgb_l      = outputs_l['image']
        if self.cfg.optim.use_SD:
            # # text embeddings
            if self.cfg.guide.append_direction:
                dirs = data['dir']  # [B,]
                text_z = self.text_z[dirs]
            else:
                text_z = self.text_z
            #### need to replace it SDedit ##################################################################
            diff_rgb = self.diffusion.train_step(text_z, pred_rgb) # noisy image ?
            #### need to replace it SDedit ##################################################################
        else:
            # diff_rgb = self.diffusion.train_step(pred_rgb, pred_rgb_mask, self.ref_image, self.ref_image_tensor, 
            #                                      use_clip=use_clip)
            diff_rgb = self.diffusion.lantent_forward(pred_rgb, pred_rgb_mask, self.ref_image, 25, rand_latent=False)
            # diff_rgb2 = self.diffusion.lantent_forward(pred_rgb, pred_rgb_mask, self.ref_image, 25, rand_latent=False)
            # rand_rgb = torch.randn_like(pred_rgb, device=self.device)
            
            # transforms.ToPILImage()(((pred_rgb[0])*pred_rgb_mask[0])[:3]).save('test.png') # latent 
            # import pdb;pdb.set_trace()
            # _rgb_mask = F.interpolate(pred_rgb_mask, (512, 512), mode='bilinear', align_corners=False)
            # transforms.ToPILImage()((diff_rgb[0])*pred_rgb_mask[0]).save('test.png')
            # transforms.ToPILImage()(diff_rgb[0]).save('test.png')
            # transforms.ToPILImage()(diff_rgb20]).save('test.png')
            
            # transforms.ToPILImage()(pred_rgb[0]).save('test.png')
            # transforms.ToPILImage()(rand_rgb[0]).save('test.png')
            
            # transforms.ToPILImage()((diff_rgb[0]*0.5 + 0.5)*_rgb_mask).save('test.png')
            # transforms.ToPILImage()((pred_rgb[0]*0.5 + 0.5)*pred_rgb_mask[0]).save('test.png')
            # import pdb;pdb.set_trace()
        
        ### TODO
        ## make CLIP loss
        # problem :: CLIP loss does not properly backprop to texture map
        # with torch.autograd.set_detect_anomaly(True):
        if use_clip:
            out_image_embeds = self.clip_image_embeddings(diff_rgb, use_grad=True)
            
            # Normalize features
            out_image_embeds = out_image_embeds / out_image_embeds.norm(dim=-1, keepdim=True)
            ref_image_embeds = self.ref_image_embeds / self.ref_image_embeds.norm(dim=-1, keepdim=True)
            
            loss_guidance = self.criterionCLIP(out_image_embeds, ref_image_embeds) * 10
        else:
            ### Pixel-wise Loss
            ref_image = self.ref_image_tensor.detach()[None]
            loss_guidance = self.criterionL1(ref_image, pred_rgb) #* self.cfg.optim.lambda_pixelwise
            # loss_guidance = self.criterionL1(ref_image[None], pred_rgb) + self.criterionL2(ref_image[None], pred_rgb)
            # loss_guidance = self.criterionL1(ref_image*pred_rgb_mask, pred_rgb*pred_rgb_mask) #* self.cfg.optim.lambda_pixelwise
        
        if log:
            save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
            # transforms.ToPILImage()(pred_rgb[0]*0.5 + 0.5).save(save_path)
            transforms.ToPILImage()(diff_rgb[0]*0.5 + 0.5).save(save_path)
            
        # import pdb; pdb.set_trace()
        # image_inputs = self.diffusion.image_processor(images=pred_rgb[0], return_tensors="pt").pixel_values
        # loss_guidance.backward()
        
        # self.mesh_model.texture_img.norm()
        # self.mesh_model.texture_img = self.mesh_model.texture_img.detach()
        # self.mesh_model.texture_img.requires_grad_(True)
        
        # loss_guidance = self.diffusion.train_step(pred_rgb, pred_rgb_mask, self.ref_image)
        # return pred_rgb, loss
        return pred_rgb, lap_loss, loss_guidance

    def train_render_text(self, data: Dict[str, Any]):
        theta    = data['theta']
        phi      = data['phi']
        radius   = data['radius']

        # import pdb;pdb.set_trace()
        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        
        ## rendered w/ current texture
        pred_rgb        = outputs['image']      # [B, 3, H, W]
        pred_rgb_mask   = outputs['mask']       # [B, 1, H, W]
        lap_loss        = outputs['lap_loss']
        
        # import pdb;pdb.set_trace()
        # pred_back = outputs['background']
        # print('{}'.format(pred_back[0,0,0]))
        
        # Guidance loss
        if self.cfg.optim.use_SD:
            # # text embeddings
            if self.cfg.guide.append_direction:
                dirs = data['dir']  # [B,]
                text_z = self.text_z[dirs]
            else:
                text_z = self.text_z
            # import pdb;pdb.set_trace()
            # loss_guidance = self.diffusion.train_step(text_z, pred_rgb, self.ref_image)
            loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        else:
            loss_guidance = self.diffusion.train_step(pred_rgb, pred_rgb_mask, self.ref_image)
        # loss = loss_guidance # dummy

        # return pred_rgb, loss
        return pred_rgb, lap_loss, loss_guidance

    def eval_render(self, data):
        theta    = data['theta']
        phi      = data['phi']
        radius   = data['radius']
        dim      = self.cfg.render.eval_grid_size
        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
                                         test=True ,dims=(dim,dim))
        pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        pred_rgb_mask = outputs['mask'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return pred_rgb, texture_rgb

    def log_train_renders(self, preds: torch.Tensor):
        if self.mesh_model.latent_mode:
            pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
            if not self.cfg.optim.use_SD:
                pred_rgb = self.diffusion.denorm_img(pred_rgb)
        else:
            pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        log_idx = self.train_step // self.cfg.log.save_interval
        # save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path = self.train_renders_path / f'step_{log_idx:05d}.jpg'
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
