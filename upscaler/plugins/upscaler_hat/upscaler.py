import os
import torch
import numpy as np

import hat.archs
import hat.data
import hat.models  # do NOT comment, it is required

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import yaml_load
from basicsr.models import build_model
from basicsr.utils import img2tensor, tensor2img

from upscaler.tools.mcm import modelhub
from upscaler.plugins.upscaler_basicsr import BASICSR

yaml_names = {
    "HAT_GAN_Real_SRx4": "HAT_GAN_Real_SRx4.yml",
    "HAT_GAN_Real_sharper": "HAT_GAN_Real_sharper.yml",
    "HAT-L_SRx2_ImageNet-pretrain": "HAT-L_SRx2_ImageNet-pretrain.yml",
    "HAT-L_SRx3_ImageNet-pretrain": "HAT-L_SRx3_ImageNet-pretrain.yml",
    "HAT-L_SRx4_ImageNet-pretrain": "HAT-L_SRx4_ImageNet-pretrain.yml"
}

modelhub_model_names = {
    "HAT_GAN_Real_SRx4": "Real_HAT_GAN_SRx4",
    "HAT_GAN_Real_sharper": "Real_HAT_GAN_sharper",
    "HAT-L_SRx2_ImageNet-pretrain": "HAT-L_SRx2_ImageNet-pretrain",
    "HAT-L_SRx3_ImageNet-pretrain": "HAT-L_SRx3_ImageNet-pretrain",
    "HAT-L_SRx4_ImageNet-pretrain": "HAT-L_SRx4_ImageNet-pretrain"
}

opt_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "options")


class HAT(BASICSR):
    def __init__(self, yaml_names=yaml_names, modelhub_model_names=modelhub_model_names, opt_folder=opt_folder,
                 model_name="HAT_GAN_Real_SRx4", tile_size=None, **kwargs):
        super(HAT, self).__init__(yaml_names, modelhub_model_names, opt_folder, model_name, tile_size, **kwargs)
