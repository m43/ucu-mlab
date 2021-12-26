import os
import pathlib

from torchvision import transforms

from corruptions.rgb_sensor_degradations import *
from utils.util import setup_torch_reproducibility

setup_torch_reproducibility(72)

img_path = "./imgs/van_gogh_room.png"
output_path_format = "./imgs/{}.png"
img = Image.open(img_path)
img = img.convert('RGB')

for corruption_name in d:
    for severity in range(1, 5 + 1):
        corrupt_img = Image.fromarray(apply_corruption(img, corruption_name, severity))
        corrupt_img.save(output_path_format.format(corruption_name.lower().replace(" ", "_") + f"_s={severity}"))

for i in range(10):
    corrupt_img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)(img)
    corrupt_img.save(output_path_format.format(f"color_jitter_random{i}"))

    corrupt_img = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))(img)
    corrupt_img.save(output_path_format.format(f"random_affine_random{i}"))

print(f"Done. Find your saved images in: {os.path.abspath(pathlib.Path(output_path_format).parent)}")
