from collections import deque
import numpy as np 
import argparse
import cv2
import time
import imutils


greenLower = np.array([110,50,50])
greenUpper = np.array([130,255,255])
vs = cv2.VideoCapture(0)
pts = deque(maxlen=64)
while True:

    _, frame = vs.read()

    if frame is None:
        break
    
    #frame = imutils.resize(frame, width=600)
    #blur = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,greenLower,greenUpper)
    mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1)

    cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    #cnts = imutils.grab_contours(cnts)

    center = None

    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"])) , (int(M["m01"] / M["m00"]))

        if radius > 5:

            cv2.circle(frame, (int(x), int(y)), int(radius),(0,255,255),2)
            cv2.circle(frame, center, 5, (0,0,255),-1)

    pts.appendleft(center)

    for i in range(1,len(pts)):

        if pts[i-1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(len(pts) / float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0,0,255),thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


vs.release()
cv2.destryAllWindows()