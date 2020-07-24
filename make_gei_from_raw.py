import argparse
import cv2
import numpy as np

DESCRIPTION = '''
    Make Gait Energy Images for a given sequence of raw video frames
'''

def make_gei_from_raw(frames, weight_thresh=1.5, hog_width=400, display=False, dump_silhouettes=False):
    '''
    Takes a series of video frames (assumed to contain no more than one human) and does the following:
        - Performs background subtraction
        - Detects human bounding box (take the highest weighted box that is at least above a threshold weight)
        - If bounding box found apply to background subtraction, resizing to 170x271
        - Construct GEI by taking mean of each pixel across the bounding boxes

    :param frames: a sequence of input frames, list(np.array[h,w,c]) or list(np.array[h,w])
    :param weight_thresh: threshold weight required for bounding boxes [1.5]
    :param hog_width: Target width resizing - default (400) is usually fine, int [400]
    :param display: if True then display each image of the sequence with the bounding box, bool [False]
    :return: GEI image, np.array([271, 170])
    '''
    if len(frames) == 0:
        print('Must be at least one frame')
        return np.empty((271, 170), dtype=np.uint8)

    h, w = frames[0].shape[:2]

    # background subtraction
    frames_bg = background_subtraction(frames)

    # human detections
    silhouettes = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    scaling = hog_width/w
    hog_height = int(h * scaling); print('hog_height={}'.format(hog_height))
    going_right = [] # used to track which direction we are going in
    last_x = 0
    for frame, frame_bg in zip(frames, frames_bg):

        frame_resized = cv2.resize(frame, (hog_width, hog_height))
        assert hog_height < h
        (rects, weights) = hog.detectMultiScale(frame_resized, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        if display:
            display_image_and_bbox(frame_resized, rects, weights)

        if len(rects) > 0:
            if weights[0] < weight_thresh:
                continue
            print('aspect: {}'.format(rects[0][2] / rects[0][3]))

            # extract and resize silhouettes
            # NB should check that the bounding box actually covers most of the subtracted figure

            # resize rects to frame_bg.shape
            x, y, bbw, bbh = [int(item/scaling) for item in rects[0]]
            silhouette = frame_bg[y:y+bbh, x:x+bbw]
            silhouette = cv2.resize(silhouette, (170, 271))
            print(silhouette.shape)
            assert silhouette.shape == (271,170)
            silhouettes.append(silhouette)

            # direction tracking
            going_right.append(x > last_x)
            last_x = x

    if dump_silhouettes:
        print('dumping silhouettes')
        show_silhouettes(silhouettes)

    # construct Gait Energy Image
    if len(silhouettes) > 0:
        gei = np.mean(np.array(silhouettes), axis=0)
        if sum(going_right) * 2 < len(going_right):  # i.e. x coord of head tended to head left
            gei = np.fliplr(gei)
    else:
        print('WARNING empty silhouettes array')
        gei = np.empty((271, 170), dtype=np.uint8)

    return gei


def background_subtraction(frames):
    '''
    Given sequence of frames apply background subtraction to each frame of the sequence and return the processed
    sequence
    :param frames: sequence of frames, [np.array((h, w, c))] or [np.array((h, w))]
    :return: frames_bg: sequence of segmented frames, type same as input
    '''
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    frames_bg = []
    for frame in frames:
        frames_bg.append(fgbg.apply(frame))
    return frames_bg


def display_image_and_bbox(img, rects, weights):
    RED = (0, 0, 255)
    for rect, weight in zip(rects, weights):
        if weight > 1.5:
            (x,y,w,h) = rect
            aspect = w/h
            cv2.rectangle(img, (x,y), (x+w, y+h), RED, 2)
            text_start = (10, 10)
            font_color = (255, 255, 255)
            text = ['aspect - {:0.2f}'.format(aspect), 'height = {}'.format(h), 'width = {}'.format(w)]
            put_image_text(img, text, text_start, font_scale=0.33, font_color=RED)
    cv2.imshow('frame and bbox', img)
    cv2.waitKey(250)


def put_image_text(img, lines, text_start, font_face=cv2.FONT_HERSHEY_DUPLEX,
                   font_scale=1., font_color=(0,0,255), thickness=1, bold_first_line=False):
    '''
    Write multiple lines of text on an image
    :param img: image to add text to, np.array([h, w, c]) or np.array([h, w])
    :param lines: list of text strings to display, [string]
    :param text_start: (x,y) coordinates the text starts from , (int, int)
    :param font_face: OpenCV font to use, cv2.Font [HERSHEY_PLAIN]
    :param font_scale: Font scaling to use (see documentation for cv2.putText(), float [1.]
    :param font_color: Open CV BGR color tuple, (int, int, int) [Red]
    :param thickness: Line thickness to use, int [1]
    :param bold_first_line: First line has bold effect by increasing font scale, bool [False]
    :return:
    '''
    loc_x = text_start[0]; loc_y = text_start[1]
    line_count = 0
    for line in lines:
        output_font_scale = font_scale
        if bold_first_line and line_count == 0:
            output_font_scale *= 1.2
        cv2.putText(img, line, (loc_x, loc_y), font_face, font_scale, font_color, thickness)
        (w, h), baseline = cv2.getTextSize(line, font_face, font_scale, thickness)
        loc_y += h + baseline
        line_count += 1

def show_silhouettes(silhouettes):
    '''
    Given an array of silhouettes to be used to generate a GEI, output each silhouette for inspection
    :param silhouettes: array of silhouettes, np.array([h, w])
    :return: None
    '''
    for s in silhouettes:
        cv2.imshow('Dump silhouettes', s)
        cv2.waitKey(100)

def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('start_frame', type=int)
    parser.add_argument('end_frame', type=int)
    parser.add_argument('--flip_vert', action='store_true', help='Flip the video vertically [False]')
    parser.add_argument('--test_mode', action='store_true', help="Just show the bounding boxes, don't save GEI image")
    parser.add_argument('--dump_silhouettes', action='store_true', help='Show each silhouette before creating GEI')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    video = cv2.VideoCapture(args.input_path);
    frames = []
    frame_count = 0
    while True:
        ok, frame = video.read()
        if not ok:
            if frame_count < args.end_frame:
                print('WARNING: read {} frames, finished before finding requested last frame {}'.format(
                    frame_count, args.end_frame))
            break

        if frame_count >= args.start_frame:
            if args.flip_vert:
                frame = np.flipud(frame)
            frames.append(frame)

        if frame_count > args.end_frame:
            break

        frame_count += 1

    gei_img = make_gei_from_raw(frames, weight_thresh=0.5, display=True, dump_silhouettes=args.dump_silhouettes)

    if not args.test_mode:
        cv2.imwrite(args.output_path, gei_img)



