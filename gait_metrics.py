import argparse
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


DESCRIPTION='''
    Given a sequence of frames, detects human pose and extracts the gait length for each frame. The maximum gait length
    is returned as the Gait Length metric.
'''

RIGHT_HIP   = 8
RIGHT_ANKLE = 10
LEFT_HIP    = 11
LEFT_ANKLE  = 13

def get_stride_angle(human):
    '''

    :param human:
    :return:
    '''
    # check we have valid keypoints
    body_parts = human.body_parts
    if np.sum([body_parts[k].score > 0. for k in body_parts.keys() if k in
            [RIGHT_HIP, RIGHT_ANKLE, LEFT_HIP, LEFT_ANKLE]]) < 4:
        return 0.
    end1 = body_parts[RIGHT_ANKLE]
    end2 = body_parts[LEFT_ANKLE]
    common_x = (body_parts[RIGHT_HIP].x + body_parts[LEFT_HIP].x)/2
    common_y = (body_parts[RIGHT_HIP].y + body_parts[LEFT_HIP].y)/2
    v1 = np.array([end1.x - common_x, end1.y - common_y])
    v2 = np.array([end2.x - common_x, end2.y - common_y])
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
    angle = angle / np.pi * 180
    return angle

def get_pose_estimator():
    model = 'mobilenet_thin'
    resolution = '432x368'
    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    return e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_path', help='path to a video')
    parser.add_argument('output_path', help='path to output the metrics')
    parser.add_argument('start', type=int, help='First frame of the sequence')
    parser.add_argument('end', type=int, help='Last frame of the sequence')
    parser.add_argument('--display', action='store_true', help='Visually display the detections')
    parser.add_argument('--display_wait_time', default=30, type=int,
                        help='wait time (ms) between display of frames [30]')
    args = parser.parse_args()

    e = get_pose_estimator()
    resize_out_ratio = 8.0
    video = cv2.VideoCapture(args.input_path)
    angles = []
    frame_count = 0
    angle = 0.
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_count += 1
        if frame_count < args.start or frame_count > args.end:
            continue
        humans = e.inference(frame, resize_to_default=True, upsample_size=resize_out_ratio)
        if args.display:
            e.draw_humans(frame, humans)
            cv2.imshow('', frame)
            cv2.waitKey(args.display_wait_time)
        if len(humans) > 0:
            angle = get_stride_angle(humans[0])
        print(angle)
        angles.append(angle)
    plt.plot(angles)
    plt.title('Stride Angle')
    plt.xlabel('Frame #')
    plt.ylabel('Stride Angle (deg)')
    plt.show()
    np.save(args.output_path, angles)