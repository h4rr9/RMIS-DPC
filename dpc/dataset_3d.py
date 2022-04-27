from torch.utils import data
from PIL import Image
from pathlib import Path

import csv
import numpy as np
import cv2
import zipfile
import torch
import io

from .utils import ToTensor


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RMIS(data.Dataset):

    def __init__(self,
                 data_path,
                 mode='train',
                 video_transforms=None,
                 last_frame_transforms=None,
                 seq_len=10,
                 num_seq=8,
                 downsample=3,
                 return_video=True,
                 return_last_frame=False):
        self.data_path = Path(data_path)
        self.mode = mode
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_video = return_video
        self.return_last_frame = return_last_frame
        self.last_frame_transforms = last_frame_transforms
        self.video_transforms = video_transforms
        self.vlen = 250

        assert self.return_last_frame or self.return_video, "dataset has to return something"
        assert self.vlen >= self.seq_len * self.num_seq * self.downsample, "Cannot sample frames! invalid sampling parameters"

        print('Using Robust Medical Instrument Segmentation data')

        if mode == 'train':
            self.data_file = self.data_path / 'data_split_train.csv'
        elif mode == 'val':
            self.data_file = self.data_path / 'data_split_val.csv'
        else:
            raise ValueError('Incorrect Mode')

        with open(self.data_file, 'r') as f:
            self.video_paths = np.array([
                row for row in csv.reader(f) if zipfile.ZipFile(
                    Path(row[0]) /
                    '10s_video.zip').infolist()  # check if zipfile is empty
                # TODO: find better way to check if zip is empty
            ]).flatten()

        print(f'Read {len(self.video_paths)} lines from {self.data_file}')

    def idx_sampler(self):

        # sample starting index
        start_idx = np.random.randint(self.vlen - self.num_seq * self.seq_len *
                                      self.downsample)
        # compute the rest of the frames
        seq_idxs = start_idx + np.arange(
            self.num_seq * self.seq_len) * self.downsample

        return seq_idxs

    def __getitem__(self, index):
        sampled_video, last_frame, mask = None, None, None

        vpath = Path(self.video_paths[index])

        data = []

        if self.return_video:

            # sample frames
            if self.return_last_frame:
                # TODO: in case of last frame, sample from end of video
                # do not sample randomly
                raise NotImplementedError()
            else:
                sampled_frame_idxs = self.idx_sampler()

            # compressed frames
            compressed_video = zipfile.ZipFile(vpath / '10s_video.zip')

            # load required frames into memory
            sampled_video = []
            for frame_idx in sampled_frame_idxs:
                compressed_frame_png = f"{frame_idx}.png"

                # read png file as bytes
                frame_buf = compressed_video.read(compressed_frame_png)

                # convert into 1-d buffer array
                # np_buf = np.frombuffer(frame_buf, np.uint8)
                frame_enc = io.BytesIO(frame_buf)

                # decode buffer into numpy-image
                # np_img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
                img = Image.open(frame_enc)

                sampled_video.append(img)

            # apply video transforms
            if self.video_transforms:
                sampled_video = self.video_transforms(sampled_video)

            # self.video transforms must contain toTensor
            # to change dims [SN * SL, H, W, C] => [SN * SL, C, H, W ]
            C, H, W = sampled_video[0].shape

            # convert into required dim order
            # [SN * SL, C, H, W] => [SN, C, SL, H, W]
            sampled_video = torch.stack(sampled_video, 0)
            sampled_video = sampled_video.view(self.num_seq, self.seq_len, C,
                                               H, W).transpose(1, 2)

            data.append(sampled_video)

        if self.return_last_frame:
            # png files
            last_frame_png = vpath / 'raw.png'
            mask_png = vpath / 'instrument_instances.png'
            

            last_frame = cv2.imread(str(last_frame_png), cv2.IMREAD_UNCHANGED)

            H, W, C = last_frame.shape
            
            mask = np.zeros(shape=(H, W), dtype=np.uint8)

            # chekc if mask exists
            if mask_png.exists():
                # read last_frame
                mask = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
                # convert to binary mask
                mask[mask > 0] = 1.
            else:
                # empty mask
                mask = np.zeros(shape=(H, W), dtype=np.uint8)
               # print("empty")
            # apply last_frame_transforms
            if self.last_frame_transforms:
               # print("imgage:", type(last_frame))
               # print("before:",mask.shape)
                last_frame, mask = self.last_frame_transforms(last_frame, mask)
               # print("after:",mask.shape)

            data.extend([last_frame, mask])

        if len(data) == 1:
            [data] = data

        return data

    def __len__(self):
        return len(self.video_paths)


if __name__ == "__main__":
    d = RMIS('/home/h4rr9/work/python/rmis/test',
             video_transforms=ToTensor(),
             downsample=3,
             seq_len=3,
             num_seq=5)
    print(len(d))
    __import__('ipdb').set_trace()
    frames = d[0]
    print(frames.shape)
