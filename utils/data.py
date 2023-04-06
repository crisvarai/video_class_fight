import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset 
from skimage.transform import resize

def get_dataframe(path):
    videos = []
    labels = []
    for folder in os.listdir(path):
        fd = path + folder + '/'
        for video in os.listdir(fd):
            vd = os.path.join(fd, video)
            i = 1 if folder == 'fights' else 0
            videos.append(vd)
            labels.append(i)
    data_dict = {
        'videos': videos,
        'labels': labels 
    }
    dataframe = pd.DataFrame(data=data_dict)
    return dataframe

def capture(filename, timesep, rgb, h, w):
    tmp = []
    frames = np.zeros((timesep, rgb, h, w), dtype=float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rval = False
    frm = resize(frame, (h, w, rgb))
    frm = np.expand_dims(frm, axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm) > 1):
        frm = frm / 255.0
    frames[i][:] = frm
    i += 1
    while i < timesep:
        tmp[:] = frm[:]
        rval, frame = vc.read()
        frm = resize(frame,( h, w, rgb))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frm = np.expand_dims(frm, axis=0)
        if(np.max(frm) > 1):
            frm = frm / 255.0
        frm = np.moveaxis(frm, -1, 1)
        frames[i-1][:] = frm
        i +=1
    return frames.astype(np.float32)

class VideoDataset(Dataset):
    def __init__(self, datas, timesep=40, rgb=3, h=160, w=160):
        self.dataloctions = datas
        self.timesep, self.rgb, self.h, self.w = timesep, rgb, h, w

    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video = capture(self.dataloctions.iloc[idx, 0], self.timesep, self.rgb, self.h, self.w)
        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(np.asarray(np.float32(self.dataloctions.iloc[idx, 1])))}
        return sample