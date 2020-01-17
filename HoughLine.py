import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(frame):
    plt.imshow(frame)
    plt.show()

img = cv2.imread('grid.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 190, 220, apertureSize=3)

kernel = np.ones((3, 3), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# minLineLength = 400
# maxLineGap = 15
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
i = 0

for line in lines:
    [rho, theta] = line[0]
    i += 1
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

import pdb; pdb.set_trace()

cv2.imshow('houlines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('houghlines5.jpg',img)
