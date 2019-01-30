#from skimage.io import imread
#from skimage.io import imsave
from scipy.misc import imresize
import numpy as np
import os
import logging
import cv2

logger = logging.getLogger("tool")

args = None


def head_finder(frame):
    '''
    Searches a binary mask image to find the centroid of the 10% of positive pixels (counting from the top row). The
    idea is this should be the head.
    :param frame: Grayscale image, np.array
    :return: (x,y) of the centroid, (int, int)
    '''
    h, w = frame.shape[:2]
    HEAD_PERCENTAGE = .1
    total_pixels = np.sum(frame)
    head_pixels = total_pixels * HEAD_PERCENTAGE

    cumul_frame = np.cumsum(frame)
    cumul_rows = cumul_frame[w::w]
    top_pct = np.where(cumul_rows > head_pixels)[0][0]

    M = cv2.moments(frame[:top_pct])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return int(cX), int(cY)


def waist_finder(frame_thresh):
    '''
    Searches a binary mask image to find the centroid of the positive pixels. This should roughly be the waist
    for a full-length body.
    :param frame: Grayscale image, np.array
    :return: (x,y) of the centroid, (int, int)
    '''
    Mtot = cv2.moments(frame_thresh[:, :])
    cXtot = int(Mtot["m10"] / Mtot["m00"])
    cYtot = int(Mtot["m01"] / Mtot["m00"])
    return cXtot, cYtot

def shift_left(img, left=10.0, is_grey=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    img_shift_left = np.zeros(img.shape)
    if left >= 0:
        if is_grey:
            img_shift_left = img[:, left:]
        else:
            img_shift_left = img[:, left:, :]
    else:
        if is_grey:
            img_shift_left = img[:, :left]
        else:
            img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0, is_grey=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """


    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    img_shift_up = np.zeros(img.shape)
    if up >= 0:
        if is_grey:
            img_shift_up = img[up:, :]
        else:
            img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            img_shift_up = img[:up, :]
        else:
            img_shift_up = img[:up, :, :]

    return img_shift_up


def shift_down(img, down=10.0):
    return shift_up(img, -down)


def load_image_path_list(path):
    """
    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg") or x.endswith("png")]
    return result


def image_path_list_to_image_pic_list(image_path_list):
    image_pic_list = []
    for image_path in image_path_list:
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_pic_list.append(im)
    return image_pic_list


def extract_human(img):
    """
    :param img: grey type numpy.array image
    :return:
    """

    left_blank = 0
    right_blank = 0

    up_blank = 0
    down_blank = 0

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        if np.sum(img[i, :]) == 0:
            up_blank += 1
        else:
            break

    for i in range(height-1, -1, -1):
        if np.sum(img[i, :]) == 0:
            down_blank += 1
        else:
            break

    for i in range(width):
        if np.sum(img[:, i]) == 0:
            left_blank += 1
        else:
            break

    for i in range(width-1, -1, -1):
        if np.sum(img[:, i]) == 0:
            right_blank += 1
        else:
            break

    img = shift_left(img, left_blank)
    img = shift_right(img, right_blank)
    img = shift_up(img, up_blank)
    img = shift_down(img, down_blank)
    return img


def center_person(img, size, method="simple"):
    """
    :param img: grey image, numpy.array datatype
    :param size: tuple, for example(120, 160), first number for height, second for width
    :param method: string, can be 'sample', or 'gravity'
    :return:
    """

    best_index = 0
    origin_height, origin_width = img.shape
    if method == "simple":
        highest = 0
        for i in range(origin_width):
            data = img[:, i]
            for j, val in enumerate(data):

                # encounter body
                if val > 0:
                    now_height = origin_height - j
                    if now_height > highest:
                        highest = now_height
                        best_index = i
                    break
    else:
        pixel_count = []
        for i in range(origin_width):
            pixel_count.append(np.count_nonzero(img[:, i]))
        count_all = sum(pixel_count)
        pixel_percent = [count * 1.0 / count_all for count in pixel_count]
        count_percent_sum = 0
        min_theta = 1
        for i, val in enumerate(pixel_percent):
            tmp = abs(0.5 - count_percent_sum)
            if tmp < min_theta:
                min_theta = tmp
                best_index = i
            count_percent_sum += val

    left_part_column_count = best_index
    right_part_column_count = origin_width - left_part_column_count - 1

    if left_part_column_count == right_part_column_count:
        return imresize(img, size)
    elif left_part_column_count > right_part_column_count:
        right_padding_column_count = left_part_column_count - right_part_column_count
        new_img = np.zeros((origin_height, origin_width + right_padding_column_count), dtype=np.int)
        new_img[:, :origin_width] = img
    else:
        left_padding_column_count = right_part_column_count - left_part_column_count
        new_img = np.zeros((origin_height, origin_width + left_padding_column_count), dtype=np.int)
        new_img[:, left_padding_column_count:] = img

    return imresize(new_img, size)


def build_GEI(img_list):
    """
    :param img_list: a list of grey image numpy.array data
    :return:
    """
    global args
    norm_width = 70
    norm_height = 210
    result = np.zeros((norm_height, norm_width), dtype=np.int)

    human_extract_list = []
    for img in img_list:
        try:
            human_extract_list.append(center_person(extract_human(img), (norm_height, norm_width)))
        except:
            logger.warning("fail to extract human from image")
    if args.viz:
        for img in human_extract_list:
            cv2.imshow('debug', img)
            cv2.waitKey(10)
    try:
        human_extract_list = np.array(human_extract_list)
        print(type(human_extract_list))
        print(human_extract_list.shape)
        print(human_extract_list[0, 80:100, 30:40])
        result = np.mean(human_extract_list, axis=0)
        print(result[80:100, 30:40])
    except:
        logger.warning("fail to calculate GEI, return an empty image")

    return result.astype(np.uint8)


def img_path_to_GEI(img_path):
    """
    convert the images in the img_path to GEI
    :param img_path: string
    :return: a GEI image
    """

    id = img_path.replace("/", "_")
    cache_file = "%s/%s_GEI.npy" % (config.Project.test_data_path, id)
    if os.path.exists(cache_file) and os.path.isfile(cache_file):
        return np.load(cache_file)
    img_list = load_image_path_list(img_path)
    img_data_list = image_path_list_to_image_pic_list(img_list)
    GEI_image = build_GEI(img_data_list)
    np.save(cache_file, GEI_image)
    return GEI_image

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--start', default=70, type=int)
    parser.add_argument('--end', default=190, type=int)
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    extract_human_img = extract_human(img)
    human_extract_center = center_person(extract_human_img, (210, 70))
    print(human_extract_center.shape)
    cv2.imshow('Centred Human Extracted', human_extract_center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_path = args.image_path.split('/')[0]
    img_list = sorted(load_image_path_list(img_path))
    img_list = img_list[args.start:args.end]
    print('img_list: {}'.format(img_list))
    img_data_list = image_path_list_to_image_pic_list(img_list)

    GEI_image = build_GEI(img_data_list) 
    cv2.imshow('GEI', GEI_image)
    cv2.waitKey(0)
