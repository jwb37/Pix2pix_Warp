from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

import torch
from PIL import Image 

import os
from pathlib import Path

ImageSuffices = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'
]

class FlowDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        self.paths = dict()
        basedir = Path(opt.dataroot, opt.phase + 'A')
        self.paths['A'] = sorted([p for p in basedir.iterdir() if p.suffix.casefold() in ImageSuffices], key=lambda p: p.name)
        basedir = Path(opt.dataroot, opt.phase + 'Flows')
        self.paths['B'] = sorted([p for p in basedir.iterdir() if p.suffix.casefold()=='.pt'], key=lambda p: p.name)

    def __getitem__(self, index):
        ans = dict()

        fname = str(self.paths['A'][index])
        ans['A_paths'] = fname

        img = Image.open(fname).convert('RGB')
        transform_params = get_params(self.opt, img.size)
        transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc==1))
        ans['A'] = transform(img)
        _, H, W = ans['A'].size()

        fname = str(self.paths['B'][index])
        ans['B_paths'] = fname
        flow = torch.load(fname)
        flow = torch.nn.functional.interpolate( flow, size=(H, W), mode='bilinear', align_corners=False )
        ans['B'] = flow.squeeze(dim=0)

        return ans

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths['A'])
