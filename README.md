# RIA Upscaler 



## Installation
```bash
pip3 install git+https://github.com/ria-com/upscaler.git
```
or
```bash
git clone https://github.com/ria-com/upscaler.git
cd ./upscaler
pip3 install -e .
```

## Quick start
```python
from upscaler import HAT as Upscaler
# from upscaler import DRCT as Upscaler

up = Upscaler(model_name="HAT_GAN_Real_SRx4", tile_size=320)

suffix = up.get_model_name()
filepath = 'npbox-323202934.jpeg'
filepath_sr = f'npbox-323202934-sr-{suffix}.jpeg'

img = cv2.imread(filepath)
sr_img = up.run(img)
ok = cv2.imwrite(filepath_sr, sr_img)
print("ok:", ok)
```

### Available HUT model names: 
- "HAT_GAN_Real_SRx4"
- "HAT_GAN_Real_sharper"
- "HAT-L_SRx2_ImageNet-pretrain"
- "HAT-L_SRx3_ImageNet-pretrain"
- "HAT-L_SRx4_ImageNet-pretrain"

### Available DRCT model names: 
- "DRCT_GAN_Real_SRx4"