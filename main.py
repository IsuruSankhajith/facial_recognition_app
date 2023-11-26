import os

import cv2
cap = cv2.videoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

imgBackground = cv2.imread('Resources/background.png')

folderModePath = 'Resourece/Modes'
modepathList = os.listdir(folderModePath)
while True:
    success, img = cap.rad()

    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:162 + 480, 55:55 + 640] = imgModeList[0]

    cv2.imshow("Face Attendance ", img)
    cv2.imshow("Face Attendance ", imgBackground)
    cv2.waitKey(1)