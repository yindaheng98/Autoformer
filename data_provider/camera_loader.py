import os
import re
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils2.timefeatures import time_features
from .data_loader import Dataset_Custom


class CameraDataset(Dataset_Custom):

    regex = re.compile(r"^([0-9.]+), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\), \(([-0-9.]+), ([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")
    fps = 90

    def __read_data__(self):
        self.scaler = StandardScaler()
        ts, Ts, quaternions = [], [], []
        with open(os.path.join(self.root_path,
                               self.data_path), 'r') as f:
            for line in f.readlines():
                find = re.findall(self.regex, line)[0]
                find = [float(f) for f in find]
                t, T, quaternion = find[0], find[1:4], find[4:8]
                ts.append(t)
                Ts.append(T)
                quaternions.append(quaternion)
        data = torch.concat((torch.tensor(Ts), torch.tensor(quaternions)), dim=-1)
        data_stamp = time_features(pd.to_datetime(np.asarray(ts) * self.fps, unit=self.freq), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
