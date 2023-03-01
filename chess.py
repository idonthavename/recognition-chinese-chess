# encoding: utf-8

import sys
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

path = './5.jpg'


if __name__ == '__main__':  # 主程序
    img = cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 内核比较大主要是为了去掉边界的干扰
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (24, 24))

    gradX = cv.morphologyEx(gray, cv.MORPH_OPEN, rectKernel)
    cv.imshow('gradX2', gradX)

    edges = cv.Canny(gradX, 50, 150, apertureSize=3)
    cv.imshow("edges", edges)

    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

    sx = zx = []

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow('houghlines3.jpg', img)

    cv.waitKey()
