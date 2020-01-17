#!/usr/local/bin/python3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


if len(sys.argv) == 1:
    print("ERROR: Needs an input image")
    sys.exit()

img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color = ('b','g','r')
col_hsv = ('hue', 'saturation', 'value')
plt.subplot(2,1,1)
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col, label=col)
    plt.xlim([0,256])
plt.legend()
plt.subplot(2,1,2)
for i, col in enumerate(col_hsv):
    histhsv = cv2.calcHist([hsv],[i], None,[256],[0,256])
    plt.plot(histhsv, label=col)
    plt.xlim([0,256])
plt.legend()
plt.ylim(0, 60000)
plt.show()
