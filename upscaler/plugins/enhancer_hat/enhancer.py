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

yaml_names = {
    "HAT_GAN_Real_SRx4": "HAT_GAN_Real_SRx4.yml"
}

modelhub_model_names = {
    "HAT_GAN_Real_SRx4": "Real_HAT_GAN_SRx4"
}


class HAT(object):
    def __init__(self, model_name="HAT_GAN_Real_SRx4", tile_size=None):
        opt_folder = os.path.dirname(os.path.abspath(__file__))
        opt = yaml_load(os.path.join(opt_folder, "options", yaml_names[model_name]))
        if tile_size is not None:
            opt['tile']['tile_size'] = tile_size
        opt['dist'] = False
        opt['rank'], opt['world_size'] = get_dist_info()
        set_random_seed(opt.get('manual_seed') + opt['rank'])
        opt['is_train'] = False
        if opt['num_gpu'] == 'auto':
            opt['num_gpu'] = torch.cuda.device_count()
        model_info = modelhub.download_model_by_name(modelhub_model_names[model_name])
        opt['path']['pretrain_network_g'] = model_info['path']
        self.model = build_model(opt)
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

    def run(self, img_lq):
        if img_lq.dtype != 'float32':
            img_lq = img_lq.astype(np.float32) / 255.

        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_lq = torch.unsqueeze(img_lq, 0)
        self.model.lq = img_lq.to(self.device)
        self.model.pre_process()
        if 'tile' in self.model.opt:
            self.model.tile_process()
        else:
            self.model.process()
        self.model.post_process()
        result = self.model.output.detach().cpu()
        sr_img = tensor2img([result])
        torch.cuda.empty_cache()
        return sr_img


