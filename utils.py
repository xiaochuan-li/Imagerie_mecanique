# *_*coding:utf-8 *_*
import os

import cv2
import gc
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

path = "D:\\download\\gr_5_test_2_pictures-20191002T093219Z-001\\gr_5_test_2_pictures"
path_save = "D:\\download\\gr_5_test_2_pictures-20191002T093219Z-001\\res1"

path_gif = "D:\\download\\gr_5_test_2_pictures-20191002T093219Z-001\\res1.gif"
path_force = "C:\\Users\\Administrator\\Desktop\\mec\\data\\gr_5_test_2_effort.csv"


def get_data(path):
    data = pd.read_csv(path, sep=',')
    # print(data)
    # print(data[" F(N)"])
    # quit()
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
    """
    optimization d'une function
    :param f: function, la function a fit
    :param x: np.array, les variables de distances
    :param y: np.array, les tensions
    :return:
    """
    popt, pcov = curve_fit(f, x, y)
    a, b, c, d, e = popt
    return f(x, a, b, c, d, e)


def func(x, a, b, c, d, e):
    """
    la function a fit
    :param x: np.array, les variables de distances
    :param a: float, coefficient
    :param b: float, coefficient
    :param c: float, coefficient
    :param d: float, coefficient
    :param e: float, coefficient
    :param f: float, coefficient
    :return: np.array, les tensions
    """
    coe = [a, b, c, d, e]
    res = np.zeros(x.shape)
    for i in range(len(coe)):
        res += np.power(x, i) * coe[i]
    return res


def copy_mat(img):
    out = np.ndarray(img.shape)
    out[:] = img
    return out


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


def get_distance(points):
    x_s = [x[0] for x in points]
    y_s = [x[1] for x in points]
    return [max(x_s) - min(x_s), max(y_s) - min(y_s)]


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
        # print(x_s, x_f, y_s, y_f)
        return img_x
    else:
        img_x = img[x_s:x_f, y_s:y_f]
        # print(x_s, x_f, y_s, y_f)
        return img_x


def max_filtre(x, ker=100):
    l = len(x)
    m = np.zeros((ker, l - ker + 1))
    for i in range(ker):
        m[i, :] = x[i:l - ker + i + 1]
    # print(np.max(m, axis=0))
    return np.max(m, axis=0)


def means_k(x):
    from sklearn.cluster import KMeans
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(x.reshape((-1, 1)))  # 聚类
    return estimator.labels_


def find_empty(img):
    img_0 = img  # .copy()
    img_0[img > 250] = 0
    img_0[img <= 250] = 255

    get_sum0 = max_filtre(np.sum(img_0, axis=0))
    get_sum1 = max_filtre(np.sum(img_0, axis=1))
    label0 = means_k(get_sum0)
    label1 = means_k(get_sum1)
    y_d, y_f = get_borne_axis(label0)
    x_d, x_f = get_borne_axis(label1)
    return (x_d, y_d), (x_f, y_f)


def get_axis_char(img, dim=0):
    single_dim = np.sum(img, axis=dim)
    var_max = np.max(single_dim) * 0.95
    single_dim[single_dim <= var_max] = 0
    single_dim[single_dim > var_max] = 1
    single_dim *= np.arange(len(single_dim), dtype='uint32')
    res = [x for x in single_dim if x > 0]
    padding = int((max(res) - min(res)) * 0.15)
    # print(max(res),min(res))
    return min(res) + padding, max(res) - padding


def exp_test(path):
    img = read_img(path)
    [x1, x2] = get_axis_char(img, 0)
    img_new = np.zeros(img.shape)
    img_new[800:1200, x1:x2] = 255 - img[800:1200, x1:x2]
    # data_0=np.sum(img_new,1)[:500]
    # data=np.sum(img_new,1)[800:1200]
    # plt.plot(np.arange(len(data_0)), data_0)
    # plt.plot(np.arange(len(data)),data)
    # plt.show()
    #quit()
    img_new[img_new <= 150] = 0
    img_new[img_new > 150] = 1
    return img_new


def pre_analyse(img):
    # show the image to analyse
    cv2.imshow("test", img)
    cv2.waitKey()

    # we have decided to analyse acording to the value of each pixel
    # it's likely that we are going to treat a problem of classification
    # witch means that we have to decide if this point is "interesting" (ROI)
    # but first of all i would like to take a look at its distribution

    img_t = 255 - img
    # attention that we take 1(white) for "there is someting" and 0(black) for
    # "there is nothing"

    img_s = np.ravel(img_t)
    plt.hist(img_s, bins=20, label="distribution of value of image")
    plt.legend()
    plt.show()
    # we notice that a pick apearing at about 0, which is reasonable cause we
    # have a pure background, so we just delete those pixel in the background

    img_show = [pix for pix in img_s if pix > 10]
    plt.hist(img_show, bins=20, label="distribution of value of image traited")
    plt.legend()
    plt.show()
    img_show[img_show <= 180] = 0
    img_show[img_show > 180] = 1
    plt.imshow(img_show, cmap="gray")


def get_distance_all(path, path_save):
    # plot_Data(path_force)
    # quit()
    if not os.path.isdir(path_save):
        os.mkdir(path_save)
    if not os.path.isdir(path):
        print("Not a dir")
        exit()
    distance = []
    for root, dir, files in os.walk(path):
        for i, img_path in enumerate(sorted(files)):
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
                cv2.imwrite(os.path.join(path_save, new_path), final_img[950:1150, 150:850] * 255)
    distance = np.array(distance, dtype=[("t", float), ("y", float), ("x", float)])
    distance["x"] /= distance["x"][0]
    distance["y"] /= distance["y"][0]
    return distance


def save_distance(data):
    new_csv = pd.DataFrame(data)
    new_csv.to_csv("data\\distance.csv")


if __name__ == "__main__":
    distance = get_data("data\\distance.csv")[:-10]
    plt.subplot(1,2,1)
    plt.title("Deformation parmis axis x et axis y en fonction du temps")
    plt.xlabel("temps")
    plt.ylabel("pixels")
    plt.plot(np.arange(len(distance)), distance["x"] - 1, 'y.', label="distance en axis x ")
    plt.plot(np.arange(len(distance)), optimize_func(func, np.arange(len(distance)), distance['x']) - 1, 'dimgray',
             label="pediction en aixs x")

    plt.plot(np.arange(len(distance)), distance["y"] - 1, 'g.', label="distance en axis y ")
    plt.plot(np.arange(len(distance)), optimize_func(func, np.arange(len(distance)), distance['y']) - 1, 'slategray',
             label="pediction en aixs y")
    plt.legend()
    # plt.show()
    plt.subplot(1, 2, 2)
    data_force = get_data(path_force)
    temps = data_force["t(s)"]
    force = data_force[" F(N)"]
    func_f = interp1d(temps, force)
    f_inter = func_f(distance["t"])
    plt.title("test 2 contraintes en function de deformation")
    plt.xlabel("deformation")
    plt.ylabel("contraintes/MPa")
    plt.plot(distance["x"] - 1, f_inter / 2.5 / 6, label='contrainte en fonction de deformation x')
    plt.plot(distance["y"] - 1, f_inter / 2.5 / 6, label='contrainte en fonction de deformation y')
    plt.legend()
    plt.show()
    generate_gif(path_save, path_gif)
    quit()
