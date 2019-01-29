import cv2
import argparse

def play_video(video_path, wait_time=100):
    '''
    Plays the video showing the frame number rounded to 10
    :param video_path: path to the video to be played, string
    :param wait_time:  Time in milliseconds to wait between each frame, int [100] 
    ''' 
    video = cv2.VideoCapture(video_path)
    fc = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = video.read()
        if not ok:
            break
        cv2.putText(frame, '{}'.format((fc // 10) * 10), (10,25), font, 1., (255,255,255))
        cv2.imshow('', frame)
        if fc == 0:
            cv2.waitKey(0)  # wait for user input
        else:
            cv2.waitKey(wait_time)
        fc += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    args = parser.parse_args()

    play_video(args.input_path)
