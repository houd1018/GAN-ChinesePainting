import numpy as np
import os
import config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


"""
make sure your real x & y contain in one image file
"""
class Pix2pix_Dataset(Dataset):
    def __init__(self, root_dir):
        # get root directory
        self.root_dir = root_dir
        # turn files in the directory into lists
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        # split input & target
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        # perform augumentation on both of them
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # perform augumentation on each of them specifically
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

    




