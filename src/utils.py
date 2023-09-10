import random
import os
from pathlib import Path

import numpy as np
import torch

def get_view_direction(elev, azim, top=30, front=0, angle=45):
    two_pi = 2*np.pi
    radd = lambda x : np.deg2rad(x) % (2*np.pi)
    
    # angle = np.deg2rad(45) % two_pi    
    # front = radd(front)
    azim  = azim % two_pi
    elev  = elev % two_pi
    
    view = torch.zeros(elev.shape[0], dtype=torch.long)
    
    # first determine by azim
    view[     (radd(front-angle) <= azim) | (azim < radd(front+angle))     ] = 0  # front 
    view[ (radd(front+180+angle) <= azim) & (azim < radd(front-angle))     ] = 1  # left side
    view[ (radd(front+180-angle) <= azim) & (azim < radd(front+180+angle)) ] = 2  # back
    view[     (radd(front+angle) <= azim) & (azim < radd(front+180-angle)) ] = 3  # right side
        
    view[elev < radd(top)]     = 4                                                # overhead
    view[elev > radd(180-top)] = 5                                                # bottom
    return view

# def get_view_direction(thetas, phis, overhead, front):
#     #                   phis [B,];          thetas: [B,]
#     # front = 0         [0, front)
#     # side (left) = 1   [front, 180)
#     # back = 2          [180, 180+front)
#     # side (right) = 3  [180+front, 360)
#     # top = 4                               [0, overhead]
#     # bottom = 5                            [180-overhead, 180]
#     res = torch.zeros(thetas.shape[0], dtype=torch.long)
#     # first determine by phis

#     # res[(phis < front)] = 0
#     res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

#     # res[(phis >= front) & (phis < np.pi)] = 1
#     res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

#     # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
#     res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

#     # res[(phis >= (np.pi + front))] = 3
#     res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
#     # override by thetas
#     res[thetas <= overhead] = 4
#     res[thetas >= (np.pi - overhead)] = 5
#     return res


def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    if tensor.min() < 0:
        tensor = (tensor * 0.5) + 0.5
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
