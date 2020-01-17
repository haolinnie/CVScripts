import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(b,g,r),-1)
            else:
                cv2.circle(img,(x,y),cv2.getTrackbarPos('width','image'),(b,g,r),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(b,g,r),-1)
        else:
            cv2.circle(img,(x,y),cv2.getTrackbarPos('width','image'),(b,g,r),-1)

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('width','image',2,20,nothing)

# bind function to window
cv2.setMouseCallback('image',draw_circle)

# main function
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:           # Esc
        break
    elif k == ord('m'):   # toggle mode
        mode = not mode
    elif k == ord('c'):   # clear
        img[:] = [0,0,0]
    elif k == ord('s'):   # save image and quit
        cv2.imwrite('drawing_out.png',img)
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')

cv2.destroyAllWindows()
