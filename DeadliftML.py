import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np

import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

window = tk.Tk()
window.geometry("480x700")
window.title("SwolyBible")
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20),padx=10,fg_color="pink")
classLabel.place(x=10,y=1)
classLabel.configure(text='STAGE')

counterLabel = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20),padx=10,fg_color="pink")
counterLabel.place(x=160,y=1)
counterLabel.configure(text='REPS')

probLabel = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20),padx=10,fg_color="pink")
probLabel.place(x=300,y=1)
probLabel.configure(text='PROB')

classBox = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20), fg_color="gray")
classBox.place(x=10,y=41)
classBox.configure(text='0')

counterBox = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20), fg_color="gray")
counterBox.place(x=160,y=41)
counterBox.configure(text='0')

probBox = ck.CTkLabel(window, height=40, width=120,text_color="white", font=("Arial",20), fg_color="gray")
probBox.place(x=300,y=41)
probBox.configure(text='0')

def reset_counter():
    global counter
    counter = 0
button = ck.CTkButton(window, text="RESET", command=reset_counter, height=40, width=120,text_color="white", font=("Arial",20), fg_color="pink")
button.place(x=10,y=600)

frame = tk.Frame(height=480,width=480)
frame.place(x=10,y=90)
lmain = tk.Label(frame) ##Updating labels pass frame through
lmain.place(x=0,y=0)

mp_drawing = mp.solutions.drawing_utils ##import mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5,min_detection_confidence=0.5)

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
current_stage='' ##Up or down during deadlift
counter = 0 ##count using ML
bodylang_prob = np.array([0,0])
bodylang_class = ''

def detect():
    global current_stage
    global counter
    global bodylang_prob
    global bodylang_class

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 3),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 3)
                              )
    
    try:
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0]

        if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "down"
        elif current_stage == "down" and bodylang_class =="up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up"
            counter += 1
    except Exception as e:  ##Where we do our detection
        print(e)
    img = image[:, :460, :]
    img_array = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_array)
    lmain.img_tk = img_tk
    lmain.configure(image=img_tk)
    lmain.after(10, detect)

    counterBox.configure(text = counter)
    probBox.configure(text = bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text = current_stage)

detect()
window.mainloop()