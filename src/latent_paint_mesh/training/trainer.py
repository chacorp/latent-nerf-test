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
from tqdm import tqdm

from src import utils
from src.latent_paint.configs.train_config import TrainConfig
from src.latent_paint.training.views_dataset import ViewsDataset
from src.stable_diffusion import StableDiffusion
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
        self.text_z             = self.calc_text_embeddings()
        self.optimizer          = self.init_optimizer()
        self.dataloaders        = self.init_dataloaders()
        
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

    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.mesh_model.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

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

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        step_weight = 0
        while self.train_step < self.cfg.optim.iters:
            
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                
                ## Laplace loss weighting
                ## ref: https://github.com/NasirKhalid24/CLIP-Mesh/blob/d3cf57ebe5e619b48e34d6f0521a31b2707ddd72/configs/paper.yml
                if self.train_step == 0:
                    laplacian_weight = self.cfg.optim.laplacian_weight
                    laplacian_min = self.cfg.optim.laplacian_min
                else:
                    laplacian_weight = (laplacian_weight - laplacian_min) * 10.**(-self.train_step*1e-06) + laplacian_min
                
                self.train_step += 1
                pbar.update(1)

                # prev_disp = self.mesh_model.displacement.clone()
                self.optimizer.zero_grad()
                self.optimizer_disp.zero_grad()

                # pred_rgbs, loss = self.train_render(data)
                pred_rgbs, lap_loss, loss_guidance = self.train_render(data)
                # sds_grad = self.mesh_model.displacement.grad
                
                # import pdb;pdb.set_trace()

                lap_loss = (lap_loss * laplacian_weight)

                ## Offset regularization
                ## ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L31
                # reg_loss = self.mesh_model.displacement.norm(dim=1).mean()
                reg_loss = torch.mean(torch.mean(self.mesh_model.displacement[None]**2, axis=1), axis=1)
                reg_loss = 10. ** self.cfg.optim.reg_weight * reg_loss # / (1 + (step_weight))
                
                loss = lap_loss + reg_loss
                loss.backward()
                # self.mesh_model.displacement.grad += sds_grad
                
                self.optimizer.step()
                self.optimizer_disp.step()
                
                # new_disp = self.mesh_model.displacement - prev_disp
                new_disp = torch.mean(self.mesh_model.displacement)
                # vert_mean = torch.mean(self.mesh_model.mesh.vertices)
                # pbar.set_description("loss: {:.06f} vert: {:.06f} disp:{:.06f}".format(loss.item(), vert_mean, new_disp))
                pbar.set_description("loss: {:06f} reg:{:06f} disp:{:06f}".format(
                    lap_loss.item(), reg_loss.item(), new_disp))
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

    def train_render(self, data: Dict[str, Any]):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']

        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius)
        pred_rgb = outputs['image']
        lap_loss = outputs['lap_loss']
        # pred_back = outputs['background']
        # print('{}'.format(pred_back[0,0,0]))
        
        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        # Guidance loss
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        # loss = loss_guidance # dummy

        # return pred_rgb, loss
        return pred_rgb, lap_loss, loss_guidance

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, decode_func=self.diffusion.decode_latents,
                                         test=True ,dims=(dim,dim))
        pred_rgb = outputs['image'].permute(0, 2, 3, 1).contiguous().clamp(0, 1)
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
