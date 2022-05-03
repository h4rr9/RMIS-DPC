from torch.utils import data
from PIL import Image
from pathlib import Path

import csv
import numpy as np
import cv2
import zipfile
import torch
import io

from dpc import ToTensor

VIDEO_LENGTH = 250


def zip_test(zip_file):
    try:
        z = zipfile.ZipFile(zip_file)
        return z.infolist() is not None and len(z.infolist()) == VIDEO_LENGTH
    except zipfile.BadZipfile:
        return False


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
        self.vlen = VIDEO_LENGTH

        assert self.return_last_frame or self.return_video, "dataset has to return something"
        assert self.vlen >= self.seq_len * self.num_seq * self.downsample, "Cannot sample frames! invalid sampling parameters"

        print('Using Robust Medical Instrument Segmentation data')

        if mode == 'train':
            self.data_file = self.data_path / 'data_split_train.csv'
        elif mode == 'val':
            self.data_file = self.data_path / 'data_split_val.csv'
        elif mode == 'test':
            self.data_file = self.data_path / 'test.csv'
        else:
            raise ValueError('Incorrect Mode')

        with open(self.data_file, 'r') as f:
            self.video_paths = np.array([
                row for row in csv.reader(f)
                if zip_test(Path(row[0]) /
                            '10s_video.zip')  # check if zipfile is empty
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
            # if returning image and target, the sample is the last frames of
            # the video
            # otherwise sample is sampled
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

            last_frame = Image.open(str(last_frame_png))

            W, H = last_frame.size

            # chekc if mask exists
            if mask_png.exists():
                # read last_frame
                mask = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
                # convert to binary mask
                mask[mask > 0] = 1.
            else:
                # empty mask
                mask = np.zeros(shape=(H, W), dtype=np.uint8)

            # mask is now a PIL image
            mask = Image.fromarray(np.uint8(mask * 255), 'L')
            # apply last_frame_transforms

            if self.last_frame_transforms:

                last_frame, mask = self.last_frame_transforms(last_frame, mask)

            data.extend([last_frame, mask])

        if len(data) == 1:
            [data] = data

        return data

    def __len__(self):
        return len(self.video_paths)


def get_data(
    return_video,
    video_transforms,
    return_last_frame,
    last_frame_transforms,
    args,
    mode,
):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'rmis':
        dataset = RMIS(mode=mode,
                       data_path=args.data_path,
                       video_transforms=video_transforms,
                       last_frame_transforms=last_frame_transforms,
                       seq_len=args.seq_len,
                       num_seq=args.num_seq,
                       downsample=args.ds,
                       return_last_frame=return_last_frame,
                       return_video=return_video)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


if __name__ == "__main__":
    d = RMIS(data_path='/mnt/disks/rmis_train/',
             video_transforms=ToTensor(),
             downsample=3,
             seq_len=3,
             num_seq=5)
    print(len(d))
    __import__('ipdb').set_trace()
    frames = d[0]
    print(frames.shape)
