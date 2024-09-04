import webbrowser
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import statistics as st
from playsound import playsound
import random
import os
import dlib

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')
# model =load_model(r'model.h5')


emotion_labels = {0:"anger", 1:"fear", 2:"happy", 3:"sadness", 4:"surprise"}  

cap = cv2.VideoCapture(0)
output = []
GR_dict={0:(0,255,0),1:(0,0,255)}
# i = 0
# while (i<=50):
#     ret, img = cap.read()
#     #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(img,1.05,7)

#     for x,y,w,h in faces:

#         face_img = img[y:y+h,x:x+w] 

#         resized = cv2.resize(face_img,(224,224))
#         reshaped=resized.reshape(1, 224,224,3)/255
#         predictions = model.predict(reshaped)

#         # find max indexed array
#         max_index = np.argmax(predictions[0])

#         emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
#         predicted_emotion = emotions[max_index]
#         output.append(predicted_emotion)
            
            
            
#         cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
#         cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
#         cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
#     i = i+1

#     cv2.imshow('LIVE', img)
#     key = cv2.waitKey(1)
#     if key == 27: 
#         cap.release()
#         cv2.destroyAllWindows()
#         break
# print(output)

# BASE_PATH = "D://Downloads/figma/shape_predictor_68_face_landmarks.dat"
# face_detector = dlib.frontal_face_detector();
# landmark_detector = dlib.shape_predictor(BASE_PATH);




# In this case, each element of the array corresponds to the predicted probability for a specific class. The first element is the probability for class 0, the second element is for class 1, and so on.

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad', 'Surprise']
# To extract the probabilities for each class, you can use array indexing. The [0] in model.predict(image)[0] is used to get the probabilities for the first instance (assuming it's a single prediction). If you are making predictions for multiple instances simultaneously (in a batch), you might have multiple rows in the output tensor, and you would iterate over those rows accordingly.

# In many cases, models are designed to predict probabilities for multiple instances at once (batch prediction), and the [0] index is used to select the probabilities for the first instance in the batch.

# If your model is a binary classification model, predicting only one class (e.g., class 0 or class 1), you might not need the [0] index, and you can directly use the predicted probability as a scalar value.
# Always check the shape of the output of model.predict(image) to understand the structure of the predictions and adjust your code accordingly. If you're uncertain, you can provide more details about your model architecture and the number of classes, and I can provide more specific guidance.



i = 0
output = []
while (i<=50):
    i+=1
    _, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # gray = np.expand_dims(frame, axis=2)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            # print(prediction)
            label=emotion_labels[prediction.argmax()]  # Here label stores the emotion detected.  
            label_position = (x,y)
            # print(label)
            output.append(label)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()



# import multiprocessing
# from playsound import playsound

# p = multiprocessing.Process(target=playsound, args=("file.mp3",))
# p.start()
# input("press ENTER to stop playback")
# p.terminate()




final_output1 = st.mode(output)
# final_output1 = st.mode(output)
print(final_output1)



emotion = final_output1
search_query = f"{emotion} mood music playlist"

# Construct the YouTube URL for the search query
youtube_url = f"https://youtube.com/search?q={search_query} "

# Open the YouTube URL in the default web browser
webbrowser.open(youtube_url)

if final_output1 =='Surprise':
    # os.startfile("moviesAngry.html")
    path= "songs/surprise"
    files=os.listdir(path)
    d=random.choice(files)
    playsound(path+"/"+d)
elif final_output1=='Sad':
    # os.startfile("moviesSad.html")
    path= "songs/sad"
    files=os.listdir(path)
    d=random.choice(files)
    playsound(path+"/"+d)
elif final_output1=='Happy':
    # os.startfile("moviesHappy.html")
    path= "songs/happy"
    files=os.listdir(path)
    d=random.choice(files)
    playsound(path+"/"+d)
elif final_output1=='fear':
    # os.startfile("moviesFear.html")
    path= "songs/fear"
    files=os.listdir(path)
    d=random.choice(files)
    playsound(path+"/"+d)
elif final_output1=='Neutral':
    # os.startfile("moviesNeutral.html")
    path= "songs/neutral"
    files=os.listdir(path)
    d=random.choice(files)
    playsound(path+"/"+d)

# cap.release()
# cv2.destroyAllWindows()