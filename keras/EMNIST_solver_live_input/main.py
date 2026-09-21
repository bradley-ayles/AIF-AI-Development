from PIL import Image, ImageDraw, ImageOps
import gradio as gr
import numpy as np
import sys

import keras
from keras import layers
from extra_keras_datasets import emnist

# Lookup table to convert EMNIST index to readable character
emnist_balanced_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't'
}

CANVAS_SIZE = 256

(trainingImages, trainingLabels), (testImages, testLabels) = emnist.load_data(type='balanced')
trainingImages = trainingImages.reshape(112800, 28 * 28) # change the shape of the tensor
trainingImages = trainingImages.astype("float32") / 256 # normalise the data

testImages = testImages.reshape(18800, 28 * 28) # change the shape of the tensor
testImages = testImages.astype("float32") / 256 # normalise the data


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(47, activation="softmax") # output layer with output size 47 (digits 0 to 9 + most letters (some lower and upper case are combined)). Softmax provides probability of each outcome being correct
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

    return emnist_balanced_mapping[int(np.argmax(prediction[0], axis = 0))] # axis is 0 because this is a rank 1 tensor (vector)


iface = gr.Interface(
    fn=sketchToImage,
    inputs=gr.Sketchpad(width = CANVAS_SIZE, height = CANVAS_SIZE, type='pil', image_mode='L', brush=gr.Brush(default_size=int(CANVAS_SIZE / 5), colors = ["black"])),
    outputs=gr.Textbox(),
)

iface.launch()