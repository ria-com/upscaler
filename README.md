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
from upscaler import DRCT as Upscaler

up = Upscaler(tile_size=320)

suffix = up.get_model_name()
filepath = 'npbox-323202934.jpeg'
filepath_sr = f'npbox-323202934-sr-{suffix}.jpeg'

img = cv2.imread(filepath)
sr_img = up.run(img)
ok = cv2.imwrite(filepath_sr, sr_img)
print("ok:", ok)
```