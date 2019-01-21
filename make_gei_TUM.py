import argparse
import numpy as np
import glob
import cv2
import gait_tools

DESCRIPTION='''
Construct Gait Energy Images (GEI) for a directory in the TUM dataset (normal walking, 
hand in pocket, etc)
'''

norm_width = 70
norm_height = 210

def make_gei_TUM(input_dir, output_dir, thresh_start, thresh_end):
    files = glob.glob('{}\\*.avi'.format(input_dir))
    print(files)
    for file in files:
        print('processing {}'.format(file))
        video = cv2.VideoCapture(file)
        gei = []
        status = '-'
        next_iteration = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            fgd_sum = np.sum(frame)
            if status == '-':
                # check if we go over the start threshold
                if fgd_sum > thresh_start:
                    status = str(next_iteration)
                    next_iteration += 1
            else:
                # check if we go below the end threshold
                if fgd_sum < thresh_end:
                    status = '-'
                    gei = np.mean(np.array(gei), axis=0).astype(np.uint8)
                    print(gei[30:40, 30:40])
                    cv2.imshow('gei', gei)
                    cv2.waitKey(0)
                    gei = []

            print('{} : {}'.format(status, fgd_sum))
            if status != '-':
                frame = cv2.medianBlur(frame, 25)
                new_img = gait_tools.center_person(gait_tools.extract_human(frame[:,:,0]), (norm_height, norm_width))
                gei.append(new_img)
                cv2.imshow(file, new_img)
                cv2.waitKey(5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input_dir', default='data\\tumiitgait\\uploaded_video\\Normal Walk',
                        help='location of a subdirectory of TUM dataset containing .PNG files')
    parser.add_argument('--output_dir', default='data\\gei\\Normal Walk')
    parser.add_argument('--threshold_start', default= 400000)
    parser.add_argument('--threshold_end', default=200000)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    thresh_start = args.threshold_start
    thresh_end   = args.threshold_end

    make_gei_TUM(input_dir, output_dir, thresh_start, thresh_end)

