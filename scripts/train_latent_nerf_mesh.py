import pyrallis

from src.latent_nerf_mesh.configs.train_config import TrainConfig
from src.latent_nerf_mesh.training.trainer import Trainer



@pyrallis.wrap()
def main(cfg: TrainConfig):
    trainer = Trainer(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.train()

if __name__ == '__main__':
    main()