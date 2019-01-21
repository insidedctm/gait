import argparse
import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DESCRIPTION = '''
    Given a series of segmentation masks in the directory vibe_out, generate a Gait Energy
    Image (GEI).
    '''

def read_frames(dir, start, end):
    file_list = glob.glob('{}/*.png'.format(dir))
    frames = []
    for f in sorted(file_list):
        print(f)
        frame = cv2.imread(f)
        frames.append(frame)
    if end < 0:
        end = len(frames)
    return frames[start:end]

def plot_statistics(frames):
    stats = []
    for frame in frames:
        frame_total = np.sum(frame) / 255.
        stats.append(frame_total)
    plt.plot(stats)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_dir', help='The directory containing a sequence of .png files')
    parser.add_argument('--start_frame', default=0, type=int, 
                        help='Frame number to start reading from (default: 0)')
    parser.add_argument('--end_frame', default=-1, type=int,
                        help='Frame number to finish reading from; negative numbers mean read up to the end (default: -1)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    frames = read_frames(args.input_dir, args.start_frame, args.end_frame)
    plot_statistics(frames)
