import cv2
import numpy
import os

haar = '/Users/jaanvikalarikkal/Desktop/open CV/lesson 9/haarcascade_frontalface_default.xml'
datasets = '/Users/jaanvikalarikkal/Desktop/open CV/lesson 9/datasets'
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir #alice, bob
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id += 1
(width, height) = (130,100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
recogniser = cv2.face.LBPHFaceRecognizer_create() #local binary pattern historgram algorithm
recogniser.train(images, labels)
face_cascade = cv2.CascadeClassifier(haar)
cam = cv2.VideoCapture(0)
while True:
    red,img = cam.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(grey, 1.3,4)






    
    cv2.putText