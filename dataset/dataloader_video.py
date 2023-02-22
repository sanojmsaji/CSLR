import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        print("")

    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)
        input_data, label = self.normalize(input_data, label)
        return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def normalize(self, video, label, file_id=None):
        if isinstance(video, list):
          video = np.array(video)
          video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
          video = torch.from_numpy(video.transpose((0, 3, 1, 2)))        
          video = video.float() / 127.5 - 1
        return video, label

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time