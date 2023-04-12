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

        self.mesh_model         = self.init_mesh_model()
        self.diffusion          = self.init_diffusion()
        ## get clip mean & std for normalization
        self.clip_mean          = torch.tensor(self.diffusion.feature_extractor.image_mean).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.clip_std           = torch.tensor(self.diffusion.feature_extractor.image_std).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.normalize_clip     = lambda x, mu, std: (x - mu) / std
        
        self.text_z             = self.calc_text_embeddings()
        # self.image_z            = self.calc_image_embeddings()
        # self.image_z, self.image= self.calc_image_embeddings()
        self.optimizer          = self.init_optimizer()
        self.dataloaders        = self.init_dataloaders()
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((512, 512)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.ref_image, self.ref_image_tensor = self.get_image()
        self.ref_pose  = self.get_reference_pose()
        self.criterionL1 = nn.L1Loss()
        self.criterionCLIP   = torch.nn.CosineSimilarity(dim=1, eps=1e-12)
        
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
    def init_diffusion(self, SD=False):
        # text-guided 
        if SD:
            diffusion_model = StableDiffusion(
                    self.device, 
                    model_name   = self.cfg.guide.diffusion_name,
                    concept_name = self.cfg.guide.concept_name,
                    latent_mode  = self.mesh_model.latent_mode
                )
            for p in diffusion_model.parameters():
                p.requires_grad  = False
            
        else:
            MODEL_NAME = 'Fantasy-Studio/Paint-by-Example'
            CACHE_DIR = "/source/skg/diffusion-project/hugging_cache"
            diffusion_model = PaintbyExample(
                    self.device, 
                    model_name  = MODEL_NAME,
                    cache_dir   = CACHE_DIR,
                    latent_mode = self.mesh_model.latent_mode
                )
            for p in diffusion_model.parameters():
                p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text]) # torch.Size([2, 77, 768]) 0: uncond 1: cond
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z
    
    def calc_image_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_image = self.cfg.guide.image
        image_z, image = self.diffusion.get_image_embeds(ref_image)
        return image_z, image[None]

    def get_image(self) -> torch.Tensor:
        image = Image.open(self.cfg.guide.image)
        # return self.transform(image)
        return image, self.transform(image)
    
    def get_reference_pose(self):
        dirs, thetas, phis, radius = circle_poses(self.device, radius=1.25, theta=1e-12, phi=0.0)
        data = {
            'dir'    : dirs,
            'theta'  : 1e-12,
            'phi'    : phis,
            'radius' : radius,
        }
        return data
    
    def optimize_text_token(self, prompt, image_features, image, max_itr=100):
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

                # data['image_z'] = self.image_z
                # pred_rgbs, loss = self.train_render(data)
                if self.train_step % 10 == 1:
                    pred_rgbs, lap_loss, loss_guidance = self.train_render_ref_view(data)
                else:
                    pred_rgbs, lap_loss, loss_guidance = self.train_render(data)
                # sds_grad = self.mesh_model.displacement.grad
                
                self.optimizer.step()
                
                # import pdb;pdb.set_trace()
                optim_step = int(50 * 10.**(-self.train_step*5e-05)+1)
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
                
                loss = lap_loss + reg_loss
                loss.backward()
                ## optimize displacement interval
                self.optimizer_disp.step()
                
                # new_disp = self.mesh_model.displacement - prev_disp
                new_disp = torch.mean(self.mesh_model.displacement)
                # vert_mean = torch.mean(self.mesh_model.mesh.vertices)
                # pbar.set_description("loss: {:.06f} vert: {:.06f} disp:{:.06f}".format(loss.item(), vert_mean, new_disp))
                descrition = ""
                
                descrition += "disp:{:06f}".format(new_disp)
                descrition += " lap: {:06f} reg:{:06f}".format(lap_loss.item(), reg_loss.item())
                # ========= use optim_step =========
                # descrition += "disp:{:06f} disp_step:{}".format(new_disp, next_optim_step)
                # if self.train_step % optim_step == 0:
                #     descrition += " lap: {:06f} reg:{:06f}".format(lap_loss.item(), reg_loss.item())
                
                pbar.set_description(descrition)
                # pbar.set_description("disp:{}".format(self.mesh_model.displacement[0]))

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.mesh_model.train()
                    step_weight +=1

                if np.random.uniform(0, 1) < 0.05:
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

    def train_render_ref_view(self, data):
        theta    = self.ref_pose['theta']
        phi      = self.ref_pose['phi']
        radius   = self.ref_pose['radius']
        dim     = self.cfg.render.eval_grid_size
        
        # outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        import pdb;pdb.set_trace()
        # theta, phi, radius   = data['theta'], data['phi'], data['radius']
        # theta, phi, radius   = self.ref_pose['theta'], self.ref_pose['phi'], self.ref_pose['radius']
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,dims=(dim,dim), ref_view=True)
        outputs = self.mesh_model.render(theta=self.ref_pose['theta'], phi=self.ref_pose['phi'], radius=self.ref_pose['radius'], decode_func=self.diffusion.decode_latents,dims=(dim,dim), ref_view=True)
        # outputs = self.mesh_model.render(theta=self.ref_pose['theta'], phi=self.ref_pose['phi'], radius=self.ref_pose['radius'], decode_func=self.diffusion.decode_latents,dims=(dim,dim), ref_view=True)
        
        ## rendered w/ current texture
        pred_rgb = outputs['image']
        pred_rgb_mask = outputs['mask']
        lap_loss = outputs['lap_loss']
        
        ### can not be used because this is not NeRF!! :: camera param, geometry from the image is unknown
        # loss_guidance = self.criterionL1(self.ref_image_tensor, pred_rgb)
        loss_guidance = 0
        
        ### clip loss
        # TODO: implement image clip loss        
        ref_img = self.diffusion.feature_extractor(images=self.ref_image, return_tensors="pt").pixel_values
        ref_img = ref_img.to(device=self.device)
        ref_image_embeddings = self.diffusion.image_encoder(ref_img)
        ### convert image 224 ,224 mean std normalize
        out_img = F.interpolate(pred_rgb, (224, 224), mode='bicubic')
        out_img = self.normalize_clip(out_img, self.clip_mean, self.clip_std)
        out_image_embeddings = self.diffusion.image_encoder(out_img)
        #### calculate cossim
                
        
        
        
        # loss_guidance = self.diffusion.train_step(pred_rgb, pred_rgb_mask, self.ref_image)
        # return pred_rgb, loss
        return pred_rgb, lap_loss, loss_guidance

    def train_render(self, data: Dict[str, Any]):
        theta    = data['theta']
        phi      = data['phi']
        radius   = data['radius']

        outputs  = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        
        ## rendered w/ current texture
        pred_rgb = outputs['image']
        pred_rgb_mask = outputs['mask']
        lap_loss = outputs['lap_loss']
        
        # import pdb;pdb.set_trace()
        # pred_back = outputs['background']
        # print('{}'.format(pred_back[0,0,0]))
        
        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z
        
        # Guidance loss
        # loss_guidance = self.diffusion.train_step(text_z, pred_rgb, self.ref_image)
        loss_guidance = self.diffusion.train_step(pred_rgb, pred_rgb_mask, self.ref_image)
        # loss = loss_guidance # dummy

        # return pred_rgb, loss
        return pred_rgb, lap_loss, loss_guidance

    def eval_render(self, data):
        theta   = data['theta']
        phi     = data['phi']
        radius  = data['radius']
        dim     = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
                                         test=True ,dims=(dim,dim))
        pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        pred_rgb_mask = outputs['mask'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_rgb = outputs['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return pred_rgb, texture_rgb

    def log_train_renders(self, preds: torch.Tensor):
        if self.mesh_model.latent_mode:
            pred_rgb = self.diffusion.decode_latents(preds).permute(0, 2, 3, 1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb = preds.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
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
