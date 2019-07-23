# Facial_Emotion_Recognition
( Automatic Facial Expression Recognition System Using Deep Network)
Facial expression recognition (FER) has been dramatically developed in recent years, thanks to the advancements in related fields, especially machine learning, image processing and human cognition

    We have used two different datasets.
      •	‘FER2013’
      •	CK+
      FER2013 is available in pixel form and each image is of 48X48 shape with pixel value varying between 0-255.
In case of CK+ dataset , images are  available in png format .Thus we have converted it into pixels and created a csv file
Method:-
We have used deep convolutional networks to classify human expression into one of seven basic human emotions. CNN includes six components: Convolutional
layer, Sub-sampling layers, Rectified linear unit (ReLU), Fully connected layer, Output layer and Softmax layer. Our general architecture consisted of CNN, consisting of multiple convolutional and dense layers. The architecture included 3 groups of 3 convolutional layers followed by a max-pool layer, and two groups of fully connected layer followed by a one final output layer.
 Since we have trained our model on both the datasets combined together. We have used opencv for live face detection using webcam and our pretrained model.
We have shown emoji also corresponding to emotion detected by model.

Result:-
We got 98% accuracy during testing on ckplus datasets and 59% on fer2013 dataset. By using both datasets combined we got testing accuracy of about 61%.
