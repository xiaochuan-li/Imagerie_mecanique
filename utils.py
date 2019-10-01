# *_*coding:utf-8 *_*
import cv2
import matplotlib.pyplot as plt
import numpy as np


class CalMask:
    def __init__(self, masks):
        self.masks = masks
        self.centers = [mask.center for mask in self.masks]
        self.distance = get_distance(self.centers)


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
        img = cv2.rectangle(img, self.p_l_t, self.p_r_b, 1)
        return img


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
        print(x_s, x_f, y_s, y_f)
        return img_x
    else:
        img_x = img[x_s:x_f, y_s:y_f]
        print(x_s, x_f, y_s, y_f)
        return img_x


def max_filtre(x, ker=100):
    l = len(x)
    m = np.zeros((ker, l - ker + 1))
    for i in range(ker):
        m[i, :] = x[i:l - ker + i + 1]
    print(np.max(m, axis=0))
    return np.max(m, axis=0)


def means_k(x):
    from sklearn.cluster import KMeans
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(x.reshape((-1, 1)))  # 聚类
    return estimator.labels_


def find_empty(img):
    img_0 = np.array(img)
    img_0[img > 250] = 0
    img_0[img <= 250] = 255

    get_sum0 = max_filtre(np.sum(img_0, axis=0))
    get_sum1 = max_filtre(np.sum(img_0, axis=1))
    label0 = means_k(get_sum0)
    label1 = means_k(get_sum1)
    y_d, y_f = get_borne_axis(label0)
    x_d, x_f = get_borne_axis(label1)
    return (x_d, y_d), (x_f, y_f)


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


if __name__ == "__main__":
    img_r = read_img(path="pic//image2.tiff")
    p1 = (800, 400)
    p2 = (1200, 800)
    img_t = crap_par_point_fix(img_r, p1, p2, False)
    img_com = np.array(img_t)
    img_t[img_t <= 100] = 1
    img_t[img_t > 100] = 0
    cv2.imshow("crap", img_t * 255)
    masks = get_all_masks(img_t)
    distance=CalMask(masks).distance
    print(distance)
    for i, mask in enumerate(masks):
        img_com = mask.get_contour(img_com)
    cv2.imshow("result", img_com * 255)
    cv2.waitKey()
