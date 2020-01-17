#!/usr/local/bin/python3
# Simple object tracking with colour
import time
import cv2
import numpy as np
import imutils
import sys

if len(sys.argv) == 1:
    VIDEO = True
    print("Camera warming up . . .")
    cap = cv2.VideoCapture(0)
    time.sleep(2)
else:
    VIDEO = False
    frame_original = cv2.imread(sys.argv[1])

cv2.namedWindow('Explorer')

# define range of blue color in HSV
hsv_yellow = [30, 255, 250]
lowerThresh = np.array([20, 50, 50])
upperThresh = np.array([40, 255, 255])

magentaLowerThresh = np.array([135, 55, 220]) ## Finger !
magentaUpperThresh = np.array([225, 225, 225])

cannyLowerThresh = 50

def nothing(x):
    pass

cv2.createTrackbar('HueMax', 'Explorer', 255, 255, nothing)
cv2.createTrackbar('SatMax', 'Explorer', 255, 255, nothing)
cv2.createTrackbar('ValMax', 'Explorer', 255, 255, nothing)

cv2.createTrackbar('HueMin', 'Explorer', 0, 255, nothing)
cv2.createTrackbar('SatMin', 'Explorer', 0, 255, nothing)
cv2.createTrackbar('ValMin', 'Explorer', 0, 255, nothing)

cv2.createTrackbar('GrayMin', 'Explorer', 0, 200, nothing)

GRAY = False

while True:
    # Take each frame
    if VIDEO:
        _, frame = cap.read()
    else:
        frame = frame_original
    frame = imutils.resize(frame, height=400)

    if GRAY:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        res1 = cv2.Canny(gray, cannyLowerThresh, 200)
    else:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image
        mask = cv2.inRange(hsv, lowerThresh, upperThresh)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask= mask)

        mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
        #creating an inverted mask to segment out the cloth from the frame
        mask2 = cv2.bitwise_not(mask1)
        #Segmenting the cloth out of the frame using bitwise and with the inverted mask
        res1 = cv2.bitwise_and(frame, frame, mask=mask1)

    cv2.putText(res1, 'Press G to toggle greyscale', (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Original', frame)
    cv2.imshow('Explorer', res1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:               # Esc
        break
    elif k == ord('g'):       # toggle gray
        GRAY = not GRAY

    upperThresh[0] = cv2.getTrackbarPos('HueMax', 'Explorer')
    upperThresh[1] = cv2.getTrackbarPos('SatMax', 'Explorer')
    upperThresh[2] = cv2.getTrackbarPos('ValMax', 'Explorer')
    lowerThresh[0] = cv2.getTrackbarPos('HueMin', 'Explorer')
    lowerThresh[1] = cv2.getTrackbarPos('SatMin', 'Explorer')
    lowerThresh[2] = cv2.getTrackbarPos('ValMin', 'Explorer')
    cannyLowerThresh = cv2.getTrackbarPos('GrayMin', 'Explorer')

cv2.destroyAllWindows()
