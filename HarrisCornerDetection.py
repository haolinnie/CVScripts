import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

filename = 'EmptyScreen.png'
img = cv2.imread(filename)
orig = img.copy()
ratio = img.shape[0]/300
img = imutils.resize(img, height = 300)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 50, 200)


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("Contours", img)
cv2.waitKey(0)

import pdb; pdb.set_trace()

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

##
bla = np.zeros(img.shape[0:2])
bla[res[:,1],res[:,0]]= 1
plt.imshow(bla)
plt.show()


import pdb; pdb.set_trace()

M = cv2.moments(bla)
contours,hierarchy = cv2.findContours(bla, 1, 2)

contours,hierarchy = cv2.findContours(thresh, 1, 2)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

y = res[:,1]
x = res[:,0]
plt.scatter(x,y)
plt.ylim([y.max(), y.min()])
plt.show()

dots = np.array(list(zip(x,y)))


import pdb; pdb.set_trace()

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
print("Threshold: ", fast.getInt('threshold'))
print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print("neighborhood: ", fast.getInt('type'))
print("Total Keypoints with nonmaxSuppression: ", len(kp))



detector = cv2.SimpleBlobDetector_create()
kpts = detector.detect(img)

##
import pdb; pdb.set_trace()
cv2.imshow('pic',img)
cv2.waitKey(0)
cv2.imwrite('subpixel5.png',img)