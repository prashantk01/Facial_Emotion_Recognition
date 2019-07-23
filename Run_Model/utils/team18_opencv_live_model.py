# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 23:14:51 2019

@author: prince
"""
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from datasets import get_labels
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
from inference import load_detection_model
from preprocessor import preprocess_input
imgf=cv2.imread('fear.jpg',1)
imgs=cv2.imread('sad.jpg',1 )
imgh=cv2.imread('happy.jpg',1 )
imgn=cv2.imread('neutral.jpg',1 )
imgd=cv2.imread('disgust.jpg',1 )
imgsu=cv2.imread('surprise.jpg',1 )
imga=cv2.imread('angry.jpg',1 )
USE_WEBCAM = True # If false, loads video file source
# parameters for loading data and images
emotion_model_path = './FER_ISMRITI/fer_ismriti/models/best_model_weights.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./FER_ISMRITI/fer_ismriti/models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./FER_ISMRITI/fer_ismriti/demo/emotion_recognition.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            cv2.imshow('image',imga)
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            cv2.imshow('image',imgs)
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            cv2.imshow('image',imgh)
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            cv2.imshow('image',imgsu)
        elif emotion_text=='fear':
            color = emotion_probability * np.asarray((200, 150, 100))
            cv2.imshow('image',imgf)
        elif emotion_text=='neutral':
            color = emotion_probability * np.asarray((182, 13, 85))
            cv2.imshow('image',imgn)
        else:
            color = emotion_probability * np.asarray((200, 200, 0))
            cv2.imshow('image',imgd)


        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()