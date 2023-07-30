import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import load_model
import cv2
from keras.optimizers import Adam


EPOCHS = 10
INIT_LR = 1e-3
BS = 16
default_image_size = tuple((256, 256))
image_size = 0
width = 256
height = 256
depth = 3

model = load_model('best_model_5conv2dense_woutBG.h5')
# model.compile(Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

# model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
disease_list = ['Bacterialblight', 'Blast', 'Brownspot', 'Healthy']


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image
        predict_disease(file_path)


def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))

    img = img / 255
    # img = np.expand_dims(img, axis=0)
    return img


def predict(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    return probabilities


def predict_disease(image_path):
    img = load_image(image_path)
    # print(img.shape())
    pr = predict(img)

    prediction = np.argmax(pr)
    percentage = pr[prediction]
    percentage = percentage*100
    predicted_disease = disease_list[prediction]
    print(predicted_disease)
    # percentage=round(percentage, 2)
    prediction_label.config(text=f'Predicted Disease: {predicted_disease}')


root = tk.Tk()
root.title('Disease Prediction')
root.geometry('800x800')

prediction_label = tk.Label(
    root, text='Rice Leaf Disease Detection', font=('Helvetica', 16))
prediction_label.pack()

label = tk.Label(root)
label.pack(pady=10)

prediction_label = tk.Label(root, text='', font=('Helvetica', 16))
prediction_label.pack()

upload_button = tk.Button(root, text='Upload Image', command=open_image)
upload_button.pack(pady=10)

root.mainloop()
