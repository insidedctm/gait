import argparse
import numpy as np
import glob
import cv2
import gait_tools
import os

DESCRIPTION='''
Construct Gait Energy Images (GEI) for a directory in the TUM dataset (normal walking, 
hand in pocket, etc)
'''

norm_width = 70
norm_height = 210

def make_gei_TUM(input_dir, output_dir, thresh_start, thresh_end):
    files = glob.glob('{}/*.avi'.format(input_dir))
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
            # negative values = 1 not 0 so set to 0
            _, frame_thresh = cv2.threshold(frame[:, :, 0], 127, 255, 0)
            fgd_sum = np.sum(frame_thresh)
            if status == '-':
                # check if we go over the start threshold
                if fgd_sum > thresh_start:
                    status = str(next_iteration)
                    next_iteration += 1
                    # track whether figure is left-to-right or right-to-left
                    last_x = 0
                    going_right = []
            else:
                # check if we go below the end threshold
                if fgd_sum < thresh_end:
                    status = '-'
                    print('gei.shape={}'.format(np.array(gei).shape))
                    gei = np.mean(np.array(gei), axis=0).astype(np.uint8)
                    if sum(going_right)*2 < len(going_right): # i.e. x coord of head tended to head left
                        gei = np.fliplr(gei)
                    #cv2.imshow('gei', gei)
                    #cv2.waitKey(0)
                    output_path = '{}/{}'.format(output_dir, get_output_filename(file, next_iteration));
                    print('outputpath = {}'.format(output_path))
                    cv2.imwrite(output_path, gei)
                    gei = []
                    last_x = 0
                    going_right = []

            print('{} : {}'.format(status, fgd_sum))
            if status != '-':
                cX, cY = gait_tools.head_finder(frame_thresh)
                top = cY - 30;
                left = cX - 85;
                bottom = cY - 30 + 270;
                right = cX + 84
                if left > 0 and right < frame.shape[1] and top > 0 and bottom < frame.shape[0]:
                    going_right.append(cX > last_x)
                    last_x = cX
                    print('{}'.format((top, left, bottom, right)))
                    print('{} : {}'.format(right+1-left, bottom+1-top))
                    assert right - left + 1 == 170
                    assert bottom + 1 - top == 271
                    print('frame_thresh.shape={}'.format(frame_thresh.shape))
                    crop = frame_thresh[top:bottom+1, left:right+1]
                    print('[{}] crop.shape={}'.format(file, crop.shape))
                    gei.append(crop)
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))
                    cv2.imshow(file, frame)
                    cv2.waitKey(15)

def get_output_filename(input_path, iteration):

    path_elems = input_path.split('/')
    filename_prefix = path_elems[-1]
    return '{}_{}.png'.format(filename_prefix, iteration)


def test_head_finder(dir, thresh_start):
    '''
    Take first movie in `dir` and read frames until the pixel sum is greater than `thresh_start`. Then
    call head_finder on the current frame.

    :param dir: Path to a folder containing 1 or more videos, string
    :param thresh_start: Only process a frame when the frame pixels sum above thresh_start, int
    '''
    file = glob.glob('{}/*.avi'.format(input_dir))[3]
    video = cv2.VideoCapture(file)
    frame_count = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        _, frame_thresh = cv2.threshold(frame[:,:,0], 127, 255, 0)

        if np.sum(frame_thresh) < thresh_start:
            frame_count += 1
            continue
        cX, cY = gait_tools.head_finder(frame_thresh)
        print('[{}] circling {}'.format(frame_count, (cX,cY)))
        cXtot, cYtot = gait_tools.waist_finder(frame_thresh)
        cv2.circle(frame, (cX, cY), 10, (0, 0, 255), 5)
        cv2.circle(frame, (cXtot, cYtot), 10, (0, 255, 0), 5)
        top = cY - 30; left = cX - 85; bottom = cY - 30 + 270; right = cX + 85
        print('{}'.format((top, left, bottom, right)))
        if left > 0 and right < frame.shape[1]:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))
            cv2.imshow(file, frame)
            keyPressed = cv2.waitKey(250)
            if keyPressed & 0xff == ord('q'):
                return
        frame_count += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--input_dir', default='data/tumiitgait/uploaded_video/Normal Walk',
                        help='location of a subdirectory of TUM dataset containing .PNG files')
    parser.add_argument('--output_dir', default='data/gei/Normal Walk')
    parser.add_argument('--threshold_start', default= 1000000)
    parser.add_argument('--threshold_end', default=200000)
    parser.add_argument('--test_mode', action='store_true', help='visually display a single sequence')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    thresh_start = args.threshold_start
    thresh_end   = args.threshold_end

    if args.test_mode:
        test_head_finder(input_dir, thresh_start)
    else:
        try:
            # Create target Directory
            os.mkdir(args.output_dir)
            print("Directory {} created".format(args.output_dir))
        except FileExistsError:
            print("Directory {} already exists".format(args.output_dir))
        make_gei_TUM(input_dir, output_dir, thresh_start, thresh_end)