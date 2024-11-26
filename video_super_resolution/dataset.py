import os
import random
import glob
import torchvision
from einops import rearrange
from torch.utils import data as data
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class PairedCaptionVideoDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            null_text_ratio=0.5,
            num_frames=16
    ):
        super(PairedCaptionVideoDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.num_frames = num_frames

        # root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder +'/lq'
            tag_path = root_folder +'/text'
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.mp4'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.mp4'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        vframes_gt, _, info = torchvision.io.read_video(filename=gt_path, pts_unit="sec", output_format="TCHW")
        fps = info['video_fps']
        vframes_gt = (rearrange(vframes_gt, "T C H W -> C T H W") / 255) * 2 - 1
        # gt = self.trandform(vframes_gt)

        lq_path = self.lr_list[index]
        vframes_lq, _, _ = torchvision.io.read_video(filename=lq_path, pts_unit="sec", output_format="TCHW")
        vframes_lq = (rearrange(vframes_lq, "T C H W -> C T H W") / 255) * 2 - 1
        # lq = self.trandform(vframes_lq)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            with open(tag_path, 'r', encoding='utf-8') as file:
                tag = file.read()

        return {"gt": vframes_gt[:, :self.num_frames, :, :], "lq": vframes_lq[:, :self.num_frames, :, :], "text": tag, 'fps': fps}

    def __len__(self):
        return len(self.gt_list)
