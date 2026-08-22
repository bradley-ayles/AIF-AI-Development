import keras
import keras.datasets.mnist

(trainingImages, trainingLabels), (lestImages, testLabels) = keras.datasets.mnist.load_data()

print(trainingImages.shape())
