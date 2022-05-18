from __future__ import division
from keras.models import model_from_json
import numpy
import numpy as np
import cv2


#loading the model
json_file = open('E:/Pre-Thesis 2/ver 1.6/facial expression & posture detection/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("E:/Pre-Thesis 2/ver 1.6/facial expression & posture detection/fer.h5")
print("Loaded model from disk")

#setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Pain', 'Happy', 'Sad', 'Fear', 'Neutral']

#
#def main():
#    cap = cv2.VideoCapture('E:/Pre-Thesis 2/ver 1.6/facial expression & Posture detection/videos/hi.jpg')
#    #emotion = None
#    while True:
#        success, img = cap.read()
#        emote = findemote(img)
#
#        
#        cv2.putText(img, str(emote), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                    (255, 0, 0), 3)
#
#        cv2.imshow("Image", img)
#        cv2.waitKey(1)

def findemote(img):
    w = 48
    h = 48
    x=None
    y=None
    emotion = None
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('E:/Pre-Thesis 2/ver 1.6/facial expression & posture detection/haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3  , 10)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= loaded_model.predict(cropped_img)
        emotion = labels[int(np.argmax(yhat))]
        #cv2.putText(img, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        #print("Emotion: "+labels[int(np.argmax(yhat))])
    return emotion


#if __name__ == "__main__":
#    main()