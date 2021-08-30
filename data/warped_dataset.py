from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

import torch
from PIL import Image 

import os
import re
from pathlib import Path

ImageSuffices = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff'
]

class WarpedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--warped_ds_input', type=str, default='both', help='desired type of input from a dataset with warps [ a | warped | both ]')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.re_strip_end = re.compile( r"(.*)_\d+(\.(?:png|jpg|jpeg))$" )

        self.paths = dict()
        for subcat in ('A', 'Warped'):
            basedir = Path(opt.dataroot, opt.phase + subcat)
            self.paths[subcat] = sorted([p for p in basedir.iterdir() if p.suffix.casefold() in ImageSuffices], key=lambda p: p.name)

        # The structure of the dataset requires spetial treatment for the 'B' folder
        basedir = Path(opt.dataroot, opt.phase + 'B')
        self.paths['B'] = [basedir / self.get_B_fname(A_path.name) for A_path in self.paths['A']]

    def get_B_fname(self, fname):
        m = self.re_strip_end.match(fname)
        if not m:
            return fname
        return ''.join(m.group(1,2))

    def __getitem__(self, index):
        ans = dict()
        for subcat in ('A', 'B', 'Warped'):
            fname = str(self.paths[subcat][index])
            ans[subcat + '_paths'] = fname

            img = Image.open(fname).convert('RGB')
            transform_params = get_params(self.opt, img.size)
            transform = get_transform(self.opt, transform_params, grayscale=(subcat != 'B'))
            ans[subcat] = transform(img)

        if self.opt.warped_ds_input == 'warped':
            ans['A'] = ans['Warped']
        elif self.opt.warped_ds_input == 'both':
            ans['A'] = torch.cat( (ans['A'], ans['Warped']), dim=0 )

        return ans

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths['A'])
