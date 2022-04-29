import os
import csv
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='/mnt/disks/rmis_test/data', type=str)

parser.add_argument('--output_file_prefix', default='data_split_', type=str)
parser.add_argument('--train_val_split', default=0.8, type=float)
parser.add_argument('--dataset_keep', default=1, type=float)

if __name__ == "__main__":

    args = parser.parse_args()
    stages = ['Stage_1', 'Stage_2', 'Stage_3']
    surgeries = ['Proctocolectomy', 'Rectal resection']
    patients = [str(i) for i in range(1, 11)]

    videos = dict()
    data_path = Path(args.data_path)

    total_videos = 0

    for g in stages:
        for s in surgeries:
            for p in patients:
                video_folder = data_path / g / s / p
                if os.path.isdir(video_folder):
    
                    videos_s_p = [
                        f.path for f in os.scandir(video_folder) if f.is_dir()
                    ]
                    videos[(
                        s,
                        p)] = videos_s_p[:int(len(videos_s_p) * args.dataset_keep)]
                    total_videos += len(videos[(s, p)])

    print(f"Total videos found ::: {total_videos}")

    # print(
    #     f"Splitting train val {args.train_val_split} : {1 -args.train_val_split}"
    # )
    

    # train_paths, val_paths = [], []

    test_paths = []

    for paths in videos.values():
        sorted_paths = sorted(paths)
        test_paths.extend(sorted_paths)

    test_paths = np.array(test_paths).reshape(-1, 1)
    
    print(f"test data size ::: {len(test_paths)}")
    
    test_csv_file = 'test.csv'

    print("Writing to csv files test.csv")

    with open(test_csv_file, 'w', encoding='UTF8',
              newline='') as test_f:

        test_writer = csv.writer(test_f)
        test_writer.writerows(test_paths)