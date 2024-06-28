import os
import torch
import numpy as np

import drct.archs
import drct.data
import drct.models  # do NOT comment, it is required

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import yaml_load
from basicsr.models import build_model
from basicsr.utils import img2tensor, tensor2img

from upscaler.tools.mcm import modelhub
from upscaler.plugins.upscaler_basicsr import BASICSR

yaml_names = {
    "DRCT_GAN_Real_SRx4": "DRCT_Real_SRx4.yml"
}

modelhub_model_names = {
    "DRCT_GAN_Real_SRx4": "Real_DRCT_mse_model"
}

opt_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "options")


class DRCT(BASICSR):
    def __init__(self, yaml_names=yaml_names, modelhub_model_names=modelhub_model_names, opt_folder=opt_folder,
                 model_name="DRCT_GAN_Real_SRx4", tile_size=None, **kwargs):
        super(DRCT, self).__init__(yaml_names, modelhub_model_names, opt_folder, model_name, tile_size, **kwargs)
