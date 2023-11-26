import os
import cv2
import face_recognition
import pickle

folderPath = 'images'
folderModePath = 'images_mode'
pathList = os.listdir(folderPath)
modepathList = os.listdir(folderModePath)
print(pathList)

imgList = []
studentIDS = []
imgModelList = []

for path in modepathList:
    imgModelList.append(cv2.imread(os.path.join(folderModePath, path)))
    print(path)
    print(os.path.splitext(path)[0])

print(studentIDS)

def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Example usage:
# findEncoding(imgModelList)
print("Encoding started ..... ")
encodeListKnown = findEncoding(imgList)
print("Encoding complete ")

# Saving the encoding list to a file using pickle
with open("EncodingFile.p", 'wb') as file:
    pickle.dump(encodeListKnown, file)

print("File saved successfully.")
