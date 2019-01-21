import cv2
import argparse

def show_bg_subtraction(video_path):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC(mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_LK)

    video = cv2.VideoCapture(video_path)
    while True:
      ok, frame = video.read()
      if not ok:
        break
  
      frame_bkg = fgbg.apply(frame)

      cv2.imshow('background', frame_bkg)
      cv2.waitKey(25)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Demonstrate background subtraction (using open-cv) on videos')
    parser.add_argument('video_path', help='Video to process')
    args = parser.parse_args()

    if args.video_path.isdigit():
        args.video_path = int(args.video_path)
    show_bg_subtraction(args.video_path)
