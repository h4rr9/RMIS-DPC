import os
import csv
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='/mnt/disks/rmis/data', type=str)

parser.add_argument('--output_file_prefix', default='data_split_', type=str)
parser.add_argument('--train_val_split', default=0.8, type=float)
parser.add_argument('--dataset_keep', default=0.1, type=float)

if __name__ == "__main__":

    args = parser.parse_args()

    surgeries = ['Proctocolectomy', 'Rectal resection']
    patients = [str(i) for i in range(1, 11)]

    videos = dict()
    data_path = Path(args.data_path)

    total_videos = 0

    for s in surgeries:
        for p in patients:
            video_folder = data_path / s / p
            if os.path.isdir(video_folder):

                videos_s_p = [
                    f.path for f in os.scandir(video_folder) if f.is_dir()
                ]
                videos[(
                    s,
                    p)] = videos_s_p[:int(len(videos_s_p) * args.dataset_keep)]
                total_videos += len(videos[(s, p)])

    print(f"Total videos found ::: {total_videos}")

    print(
        f"Splitting train val {args.train_val_split} : {1 -args.train_val_split}"
    )

    train_paths, val_paths = [], []

    for paths in videos.values():
        split_idx = int(len(paths) * args.train_val_split)
        sorted_paths = sorted(paths)
        train_paths_split = sorted_paths[:split_idx]
        val_paths_split = sorted_paths[split_idx:]

        train_paths.extend(train_paths_split)
        val_paths.extend(val_paths_split)

    train_paths, val_paths = np.array(train_paths).reshape(
        -1, 1), np.array(val_paths).reshape(-1, 1)

    print(f"train spilt size ::: {len(train_paths)}")
    print(f"val spilt size ::: {len(val_paths)}")

    train_csv_file = f'{args.output_file_prefix}train.csv'
    val_csv_file = f'{args.output_file_prefix}val.csv'

    print(f"Writing to csv files {train_csv_file} {val_csv_file}")

    with open(train_csv_file, 'w', encoding='UTF8',
              newline='') as train_f, open(val_csv_file,
                                           'w',
                                           encoding='UTF8',
                                           newline='') as val_f:

        train_writer = csv.writer(train_f)
        train_writer.writerows(train_paths)

        val_writer = csv.writer(val_f)
        val_writer.writerows(val_paths)
