import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
from einops import rearrange, repeat
from torch.distributions.multivariate_normal import MultivariateNormal
import math


class soft_robot_dataloader(Dataset):
    # Basic Instantiation
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        if self.mode == "train":
            self.dataset_path = self.args.train.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.train.num_ensemble
        elif self.mode == "test":
            self.dataset_path = self.args.test.data_path
            self.parent_path = "/tf/datasets/"
            self.num_ensemble = self.args.test.num_ensemble
        self.dataset = pickle.load(open(self.dataset_path, "rb"))
        self.dataset_length = len(self.dataset)
        self.dim_x = self.args.train.dim_x
        self.dim_z = self.args.train.dim_z
        self.dim_a = self.args.train.dim_a
        self.win_size = self.args.train.win_size

    def process_image(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0
        return img_array

    def process_depth(self, img_path):
        img_array = cv2.imread(img_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = np.float32(img_array) / 255
        return img_array

    def get_data(self, idx):
        # the gt input to the model
        gt_input = []
        for i in range(self.win_size):
            tmp = self.dataset[idx + i][2]
            gt_input.append(tmp)
        gt_input = np.array(gt_input)
        gt_input = gt_input + np.random.normal(0, 0.5, gt_input.shape)
        gt_input = torch.tensor(gt_input, dtype=torch.float32)

        # get the gt
        gt = self.dataset[idx + self.win_size][2]

        # obs rbg img
        img_index = self.dataset[idx + self.win_size][1]
        img_string = str(img_index).zfill(6)
        img_path = self.parent_path + "small_rgb/rgb-" + img_string + ".jpg"
        rgb = self.process_image(img_path)

        # obs depth img
        img_path = self.parent_path + "depth_small/rgb-" + img_string + ".png"
        depth = self.process_depth(img_path)

        # obs IMUs
        obs = np.array(self.dataset[idx + self.win_size][3])

        # action
        action = np.array(self.dataset[idx + self.win_size][4])

        # convert to tensor
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = rearrange(obs, "(k dim) -> k dim", k=1)

        action = torch.tensor(action, dtype=torch.float32)
        action = rearrange(action, "(k dim) -> k dim", k=1)
        action = repeat(action, "k dim -> n k dim", n=self.num_ensemble)
        action = rearrange(action, "n k dim -> (n k) dim")

        gt = torch.tensor(gt, dtype=torch.float32)
        gt = rearrange(gt, "(k dim) -> k dim", k=1)

        # images
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = rearrange(rgb, "h w ch -> ch h w")

        depth = torch.tensor(depth, dtype=torch.float32)
        depth = rearrange(depth, "(ch h) w -> ch h w", ch=1)

        out = (gt_input, action, obs, rgb, depth, gt)
        return out

    # Length of the Dataset
    def __len__(self):
        # self.dataset_length = 50
        return self.dataset_length - self.win_size - 5

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        # make sure always take the data from the same sequence
        not_valid = True
        while not_valid:
            try:
                if self.dataset[idx][0] == self.dataset[idx + self.win_size + 4][0]:
                    not_valid = False
                else:
                    idx = random.randint(0, self.dataset_length)
            except:
                idx = random.randint(0, self.dataset_length)

        # data from t-1
        out_1 = self.get_data(idx)

        # data from t
        idx = idx + 1
        out_2 = self.get_data(idx)

        return out_1, out_2
