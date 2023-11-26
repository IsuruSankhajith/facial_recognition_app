import os
import cv2
import face_recognition
import pickle
import numpy as np
import cvzone

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

imgBackground = cv2.imread('Resources/background.png')

folderModePath = 'Resources/Modes'
modepathList = os.listdir(folderModePath)
imgModelList = []

for path in modepathList:
    imgModelList.append(cv2.imread(os.path.join(folderModePath, path)))

file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()

# Assuming encodeListKnownWithIds contains both encoding and student IDs
encodeListKnown, studentIds = zip(*encodeListKnownWithIds)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:162 + 480, 808:808 + 414] = imgModelList[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("FaceDis", faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = (55 + x1, 162 + y1, 162 + x2, 162 + y2)
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    cv2.imshow("Face Attendance", img)
    cv2.imshow("Face Attendance with Background", imgBackground)
    cv2.waitKey(1)
