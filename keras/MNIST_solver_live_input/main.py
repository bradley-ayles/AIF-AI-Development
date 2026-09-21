from PIL import Image, ImageDraw, ImageOps
import gradio as gr
import numpy as np
import sys

import keras
from keras import layers
from keras.datasets import mnist

CANVAS_SIZE = 256

(trainingImages, trainingLabels), (testImages, testLabels) = keras.datasets.mnist.load_data()
trainingImages = trainingImages.reshape(60000, 28 * 28) # change the shape of the tensor
trainingImages = trainingImages.astype("float32") / 256 # normalise the data

testImages = testImages.reshape(10000, 28 * 28) # change the shape of the tensor
testImages = testImages.astype("float32") / 256 # normalise the data


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax") # output layer with output size 10 (digits 0 to 9). Softmax provides probability of each outcome being correct
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

History = model.fit(trainingImages, trainingLabels, batch_size=128, epochs=5)

print(History.history)

print("Testing model:")

loss = model.evaluate(testImages, testLabels)

print(f"Loss: {loss}")

print("Now it's your turn")

def sketchToImage(image):
    im2 = image['composite']
    im2 = im2.convert("L")
    im2 = ImageOps.invert(im2)
    im2 = im2.resize((28, 28))
    im2.save("img.png")

    npArray = np.array(im2) # convert the image to a numpy array
    npArray = npArray.reshape(1, 28 * 28) # change the shape of the tensor
    npArray = npArray.astype("float32") / 256 # normalise the data

    prediction = model.predict(npArray)

    # print(npArray)
    print(prediction)

    return int(np.argmax(prediction[0], axis = 0)) # axis is 0 because this is a rank 1 tensor (vector)


iface = gr.Interface(
    fn=sketchToImage,
    inputs=gr.Sketchpad(width = CANVAS_SIZE, height = CANVAS_SIZE, type='pil', image_mode='L', brush=gr.Brush(default_size=int(CANVAS_SIZE / 5), colors = ["black"])),
    outputs=gr.Textbox(),
)

iface.launch()