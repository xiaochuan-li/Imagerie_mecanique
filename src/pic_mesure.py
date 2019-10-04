# *_*coding:utf-8 *_*
import os
from optparse import OptionParser

import cv2
import numpy as np

point = []
points = []
flag = 0


def check_empty():
    global flag
    if flag == 0:
        return True
    else:
        return False


def check_full(scale, is_log=False, log_path="log.txt"):
    global point, flag
    if flag == 4:
        if is_log:
            print(log_path)
            with open(log_path, 'a') as f:
                string_log = ''
                for x in [(scale[0] * p[0], scale[1] * p[1]) for p in point]:
                    string_log += str(x[0]) + " , " + str(x[1]) + " | "
                f.write(string_log + '\n')
        points.append([(scale[0] * p[0], scale[1] * p[1]) for p in point])
        point = []
        flag = 0
        return True
    else:
        return False


def draw_circle(event, x, y, flags, param):
    global point, points, flag
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("ajoute point [{},{}] with flag = {}".format(x, y, flag + 1))
        point.append((x, y))
        flag += 1

    elif event == cv2.EVENT_RBUTTONDOWN:
        if check_empty():
            print("no point to delete")
        else:
            print("delete the point {}".format(point.pop()))
            flag -= 1


def draw(img):
    new_img = np.array(img)
    global point
    for p in point:
        new_img = cv2.circle(new_img, p, 3, (0, 0, 255))
    return new_img


def get_args():
    parser = OptionParser()
    parser.add_option('--img_path', dest='path',
                      default="D:\\download\\gr_5_test_2_pictures-20191002T093219Z-001\\gr_5_test_2_pictures",
                      type='str',
                      help='path of images')
    parser.add_option('--log_path', dest='log_path', default="log.txt",
                      type='str', help='path of log file')
    parser.add_option('--is_log', dest='is_log', action='store_true', default=False,
                      help='whether generate log file or not')
    parser.add_option('--x_length', dest='x_length', default=1024, type='int',
                      help='whether generate log file or not')
    parser.add_option('--y_length', dest='y_length', default=1024, type='int',
                      help='whether generate log file or not')
    """parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    """
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    path = args.path
    log_path = args.log_path
    new_shape = (args.x_length, args.y_length)
    is_log = args.is_log

    for root, dir, files in os.walk(path):
        for img_path in files[:2]:
            cv2.namedWindow(img_path)
            cv2.setMouseCallback(img_path, draw_circle)
            img_ori = cv2.imread(os.path.join(path, img_path))
            shape_ori = img_ori.shape
            scale = (float(shape_ori[0]) / new_shape[0], shape_ori[1] / new_shape[1])
            img = cv2.resize(img_ori, new_shape)
            while True:
                img_drawable = draw(img)
                cv2.imshow(img_path, img_drawable)
                if cv2.waitKey(20) & 0xFF == 27 or check_full(scale, log_path=log_path, is_log=is_log):
                    break
            cv2.destroyAllWindows()
    print(points)
