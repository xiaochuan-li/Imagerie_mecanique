# *_*coding:utf-8 *_*
import gc
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import config

name = config['name']
path = config['path']
path_save = config['path_save']
path_force = config['path_force']
path_gif = config['path_gif']
ratio_max = config['ratio_max']
padding_max = config['padding']
seuil_max = config['seuil']
path_distance = config['path_distance']


class Mask:
    def __init__(self, img, p1, p2):
        x_s = min(p1[0], p2[0])
        x_f = max(p1[0], p2[0])
        y_s = min(p1[1], p2[1])
        y_f = max(p1[1], p2[1])
        self.p_l_t = (y_s, x_s)
        self.p_l_b = (y_f, x_s)
        self.p_r_t = (y_s, x_f)
        self.p_r_b = (y_f, x_f)
        self.mask = crap_par_point_fix(img, self.p_l_t, self.p_r_b, False)
        self.center = (0.5 * (x_s + x_f), 0.5 * (y_s + y_f))

    def get_contour(self, img):
        img_output = copy_mat(img)
        img_output = cv2.rectangle(img_output, self.p_l_t, self.p_r_b, 1)
        new_out = copy_mat(img_output)
        del img_output
        gc.collect()
        return new_out


class ImgWithMasks:
    def __init__(self, img, masks):
        self.img = copy_mat(img)
        self.masks = masks
        self.centers = [mask.center for mask in self.masks]
        self.distance = get_distance(self.centers)

    def draw_masks(self):
        img_out = copy_mat(self.img)
        for mask in self.masks:
            img_out = copy_mat(mask.get_contour(img_out))
        return img_out


def get_data(path):
    data = pd.read_csv(path, sep=',')
    return data


def plot_Data(path):
    data = get_data(path)
    plt.plot(data["t(s)"], data[" F(N)"], label="FORCE EN FONCTION DU TEMPS")
    plt.legend()
    plt.show()


def generate_gif(path, path_save):
    frames = []
    for root, dir, imgs in os.walk(path):
        for img in imgs:
            try:
                frames.append(imageio.imread(os.path.join(path, img)))
            except BaseException:
                print("error")
    imageio.mimsave(path_save, frames, 'GIF', duration=0.1)

    return 0


def optimize_func(f, x, y):
    popt, pcov = curve_fit(f, x, y)
    a, b, c, d, e = popt
    return f(x, a, b, c, d, e)


def func_D_S(x, a, b, c, d, e):
    coe = [a, b, c, d, e]
    res = np.zeros(x.shape)
    for i in range(len(coe)):
        res += np.power(x, i) * coe[i]
    return res


def func(x, a, b, c, d, e):
    coe = [a, b, c, d, e]
    res = np.zeros(x.shape)
    for i in range(len(coe)):
        res += np.power(x, i) * coe[i]
    return res


def copy_mat(img):
    out = np.ndarray(img.shape)
    out[:] = img
    return out


def get_distance(points):
    x_s = [x[0] for x in points]
    y_s = [x[1] for x in points]
    return [max(x_s) - min(x_s), max(y_s) - min(y_s)]


def generate_mask(img, xy):
    x = xy[1]
    y = xy[0]
    return Mask(img, (x[0], y[0]), (x[1], y[1]))


def split_img(img, dim=0):
    get_sum = np.sum(img, axis=dim)
    get_sum[get_sum > 0] = 1

    a = np.array(get_sum[:-1], dtype="int32")
    b = np.array(get_sum[1:], dtype='int32')
    c = np.arange(get_sum.size - 1, dtype='int32')

    delta_sum = (a - b) * c
    mask_x_d = [-x for x in delta_sum if x < 0]
    mask_x_f = [x + 1 for x in delta_sum if x > 0]
    masks_x = list(zip(mask_x_d, mask_x_f))
    (x_d, x_f) = masks_x[1]
    return get_half_img(img, x_d, x_f, axis=0)


def get_half_img(img, x_d, x_f, axis=0):
    img_x = np.zeros(img.shape)
    if axis == 0:
        img_x[:, x_d:x_f] = img[:, x_d:x_f]
        img_y = img - img_x
    else:
        img_x[x_d:x_f, :] = img[x_d:x_f, :]
        img_y = img - img_x
    return img_y, img_x


def get_sous_masks(img, dim=0):
    mask_list = get_borne(img, 1 - dim)
    return mask_list[0]


def get_borne(img, dim=0):
    get_sum = np.sum(img, axis=dim)
    get_sum[get_sum > 0] = 1
    delta_sum = (get_sum[:-1] - get_sum[1:]) * np.arange(get_sum.size - 1)
    mask_x_d = [int(-x) for x in delta_sum if x < 0]
    mask_x_f = [int(x + 1) for x in delta_sum if x > 0]
    return list(zip(mask_x_d, mask_x_f))


def get_borne_axis(x):
    delta_x = (x[:-1] - x[1:]) * np.arange(x.size - 1)
    mask_x_d = [int(-x) for x in delta_x if x < 0]
    mask_x_f = [int(x + 1) for x in delta_x if x > 0]
    return min(mask_x_d[0], mask_x_f[0]), max(mask_x_d[0], mask_x_f[0])


def get_masks(img, dim=0):
    masks_x = get_borne(img, dim)
    if dim == 0:
        points = [(x, get_sous_masks(get_half_img(img, int(x[0]), int(x[1]), 0)[1], 0)) for x in masks_x]
    else:
        points = [(get_sous_masks(get_half_img(img, int(x[0]), int(x[1]), 1)[1], 1), x) for x in masks_x]
    return points


def get_all_masks(img):
    img_1, img_2 = split_img(img)
    p_0 = get_masks(img_1, 0)
    p_1 = get_masks(img_2, 1)
    for p in p_1:
        p_0.append(p)
    masks_list = [generate_mask(img, xy) for xy in p_0]
    return masks_list


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_cor(x, y):
    return np.correlate(x, y)


def crap_par_point_fix(img, point_1, point_2, padding=True):
    x_s = min(point_1[0], point_2[0])
    x_f = max(point_1[0], point_2[0])
    y_s = min(point_1[1], point_2[1])
    y_f = max(point_1[1], point_2[1])
    if padding:
        img_x = np.zeros(img.shape)
        img_x[x_s:x_f, y_s:y_f] = img[x_s:x_f, y_s:y_f]
        return img_x
    else:
        img_x = img[x_s:x_f, y_s:y_f]
        return img_x


def get_axis_char(img, dim=0):
    global ratio_max, padding_max
    single_dim = np.sum(img, axis=dim)
    var_max = np.max(single_dim) * ratio_max
    single_dim[single_dim <= var_max] = 0
    single_dim[single_dim > var_max] = 1
    single_dim *= np.arange(len(single_dim), dtype='uint32')
    res = [x for x in single_dim if x > 0]
    padding = int((max(res) - min(res)) * padding_max)
    # print(max(res),min(res))
    return min(res) + padding, max(res) - padding


def exp_test(path):
    global seuil_max
    img = read_img(path)
    [x1, x2] = get_axis_char(img, 0)
    img_new = np.zeros(img.shape)
    img_new[800:1200, x1:x2] = 255 - img[800:1200, x1:x2]
    cv2.blur(img, (20, 20))
    img_new[img_new <= seuil_max] = 0
    img_new[img_new > seuil_max] = 1
    return img_new


def get_distance_all(path, path_save):
    if not os.path.isdir(path_save):
        os.mkdir(path_save)
    if not os.path.isdir(path):
        print("Not a dir")
        exit()
    distance = []
    for root, dir, files in os.walk(path):
        for i, img_path in enumerate(sorted(files)[:200]):
            time = float(img_path[11:-5])
            img_trated = exp_test(os.path.join(path, img_path))

            try:
                img_with_mask = ImgWithMasks(img_trated, get_all_masks(img_trated))
            except IndexError:
                print("{}th picture --- No points detected in {}".format(str(i).zfill(3), img_path))
            except BaseException:
                print("Error in detection")
            else:
                final_img = img_with_mask.draw_masks()
                print("{}th picture --- Points detection finished with x : {} | y : {}".format(str(i).zfill(3),
                                                                                               img_with_mask.distance[
                                                                                                   0],
                                                                                               img_with_mask.distance[
                                                                                                   1]))
                distance.append((time, img_with_mask.distance[0], img_with_mask.distance[1]))
                new_path = img_path[:-5] + '.png'
                # cv2.imshow("test_fin", final_img[950:1150, :] * 255)
                # cv2.waitKey(1)
                cv2.imwrite(os.path.join(path_save, new_path), final_img[950:1150, 150:850] * 255)
    distance = np.array(distance, dtype=[("t", float), ("y", float), ("x", float)])
    distance["x"] /= distance["x"][0]
    distance["y"] /= distance["y"][0]
    save_distance(distance)
    return distance


def load_pos(path="data\\gr_5_test_2_position.csv"):
    pos = pd.read_csv(path)
    return pos


def save_distance(data, ):
    global path_distance
    new_csv = pd.DataFrame(data)
    new_csv.to_csv(path_distance)


def D_T(data_path='data/test1_distance.csv', name='test1'):
    data = get_data(data_path)[:130]
    x = data['t']
    d_x = data['x']
    d_y = data['y']
    plt.title(name + ' deformation en fonction du temps')
    plt.plot(x, d_x, '.', label='Longitudinale')
    plt.plot(x, d_y, '.', label='Transversale')
    plt.legend()
    plt.xlabel('temps(s)')
    plt.ylabel('deformation(mm/mm)')
    plt.savefig(os.path.join('data', name + '_defome-temps.png'))
    plt.show()
