import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
# img = cv2.line(img,(0,0),(511,511),(255,0,0),5)


# Draw a rectangle
# img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

#  Draw a circle
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)

# Write text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
