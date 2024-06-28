import cv2
import os
# from upscaler import HAT as Upscaler
from upscaler import DRCT as Upscaler

if __name__ == '__main__':
    up = Upscaler(tile_size=320)

    suffix = up.get_model_name()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, '../data/examples/lq/npbox-323202934.jpeg')
    filepath_sr = os.path.join(current_dir, f'../data/examples/lq/npbox-323202934-sr-{suffix}.jpeg')

    img = cv2.imread(filepath)
    sr_img = up.run(img)
    ok = cv2.imwrite(filepath_sr, sr_img)
    print("ok:", ok)
