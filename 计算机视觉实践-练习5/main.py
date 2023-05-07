import os
from glob import glob

from pylab import *

np.seterr(divide='ignore', invalid='ignore')
from PIL import Image
import numpy as np
from scipy import ndimage


def plane_sweep_ncc(im_l, im_r, start, steps, wid):
    """ 使用归一化的互相关计算视差图像 该函数返回每个像素的最佳视差"""
    m, n = im_l.shape
    # 保存不同求和值的数组
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))
    # 保存深度平面的数组
    dmaps = np.zeros((m, n, steps))
    # 计算图像块的平均值
    ndimage.filters.uniform_filter(im_l, wid, mean_l)
    ndimage.filters.uniform_filter(im_r, wid, mean_r)
    # 归一化图像
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # 尝试不同的视差
    for displ in range(steps):
        # 将左边图像移动到右边，计算加和
        ndimage.filters.uniform_filter(np.roll(norm_l, -displ - start) * norm_r, wid, s)  # 和归一化
        ndimage.filters.uniform_filter(np.roll(norm_l, -displ - start) * np.roll(norm_l, -displ - start), wid, s_l)
        ndimage.filters.uniform_filter(norm_r * norm_r, wid, s_r)  # 和反归一化
        # 保存 ncc 的分数
        dmaps[:, :, displ] = s / np.sqrt(s_l * s_r)
    # 为每个像素选取最佳深度
    return np.argmax(dmaps, axis=2)


def plane_sweep_gauss(im_l, im_r, start, steps, wid):
    """ 使用带有高斯加权周边的归一化互相关计算视差图像 """
    m, n = im_l.shape
    # 保存不同加和的数组
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))
    # 保存深度平面的数组
    dmaps = np.zeros((m, n, steps))
    # 计算平均值
    ndimage.filters.gaussian_filter(im_l, wid, 0, mean_l)
    ndimage.filters.gaussian_filter(im_r, wid, 0, mean_r)
    # 归一化图像
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # 尝试不同的视差
    for displ in range(steps):
        # 将左边图像移动到右边，计算加和
        ndimage.filters.gaussian_filter(np.roll(norm_l, -displ - start) * norm_r, wid, 0, s)  # 和归一化
        ndimage.filters.gaussian_filter(np.roll(norm_l, -displ - start) * np.roll(norm_l, -displ - start), wid, 0, s_l)
        ndimage.filters.gaussian_filter(norm_r * norm_r, wid, 0, s_r)  # 和反归一化
        # 保存 ncc 的分数
        dmaps[:, :, displ] = s / np.sqrt(s_l * s_r)
    # 为每个像素选取最佳深度
    return np.argmax(dmaps, axis=2)



# 读取左右图像
datapath = './data'
datasets = glob(os.path.join(datapath, '*'))
for dataset in datasets:
    _classes = glob(os.path.join(dataset, '*'))
    for _class in _classes:
        images = glob(os.path.join(_class, '*.png'))
        """载入图像，并使用该函数计算偏移图"""
        im_l = np.array(Image.open(images[0]).convert('L'), 'f')
        im_r = np.array(Image.open(images[1]).convert('L'), 'f')
        figure()
        gray()
        subplot(221)
        imshow(im_l)
        subplot(222)
        imshow(im_r)

        # 开始偏移，并设置步长
        steps = 12
        start = 4
        # ncc 的宽度
        wid = 7
        res = plane_sweep_ncc(im_l, im_r, start, steps, wid)
        wid2 = 7
        gauss = plane_sweep_gauss(im_l, im_r, start, steps, wid2)
        subplot(223)
        imshow(res)
        subplot(224)
        imshow(gauss)

        show()

