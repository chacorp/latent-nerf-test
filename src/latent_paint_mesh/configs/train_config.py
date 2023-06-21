from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Render height,width for training
    # train_grid_size: int = 64
    train_grid_size: int = 128
    # Render height,width for evaluation
    eval_grid_size: int = 512
    # training camera radius range
    # radius_range: Tuple[float, float] = (1.45, 1.75)
    radius_range: Tuple[float, float] = (0.8, 1.0)
    # Set [0,angle_overhead] as the overhead region
    angle_overhead: float = 40
    # Define the front angle region
    angle_front: float = 70
    # Which NeRF backbone to use
    backbone: str = 'texture-mesh'

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str
    image: str
    # The mesh to paint
    shape_path: str
    # Append direction to text prompts
    append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    # diffusion_name: str = 'CompVis/stable-diffusion-v1-4'
    # diffusion_name: str = 'runwayml/stable-diffusion-v1-5'
    diffusion_name: str = '/source/kseo/hugging_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819'
    # diffusion_name: str = '/source/kseo/huggingface_cache/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a'
    guidance_scale: float = 7.5
    
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 0.6
    # height of mesh
    dy: float = 0.25
    # dy: float = 0.7
    
    # texture image resolution
    texture_resolution=128
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str= 'nearest'


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Total iters
    # iters: int = 1000
    iters: int = 3000
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None
    
    ## Diffusion model
    use_SD: bool = False # if false use Paint-by-Example, if true use stable-diffusion
        
    ## Laplacian weight
    # ref: https://github.com/NasirKhalid24/CLIP-Mesh/blob/d3cf57ebe5e619b48e34d6f0521a31b2707ddd72/configs/paper.yml
    laplacian_weight: float = 100
    laplacian_min: float = 0.6
    
    ## loss
    lambda_pixelwise: float = 1.0
    
    
    ## Texture Learning rate
    lr: float = 1e-2
    ## Displacement
    # ref: https://github.com/bharat-b7/LoopReg/blob/ab349cc0e1a7ac534581bd7a9e30e08ce10e7696/fit_SMPLD.py#L57
    # disp_lr: float = 5e-3
    disp_lr: float = 1e-4
    lap_weight: float = 10.
    reg_weight: float = 2.


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # How many steps between save step
    save_interval: int = 500
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Export a mesh
    save_mesh: bool = True
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 2

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)

    def __post_init__(self):
        if self.log.eval_only and (self.optim.ckpt is None and not self.optim.resume):
            logger.warning('NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True

