# Simple AI model to predict future temperatures based on past temperatures

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

import csv

# constants
data_file_name = "Mt. Lofty weather data.csv"

num_days_per_sample = 6 # 7 days per sample
num_epochs = 15

test_data_split_location = 0.80 # 80% of the data is training data
validation_data_split_location = 0.90 # 10% of the data is test data

print("Reading data from file")

daily_temps = [] # list to store temps

# Open the CSV file
with open(data_file_name, mode='r') as file:
    reader = csv.DictReader(file)

    # Iterating over each row
    for row in reader:
        daily_temps.append(float(row['Maximum Temperature'].replace(",", "")))

tmp_daily_temps = daily_temps
daily_temps = []

# Convert the data into temperature difference between consecutive days (e.g. [1, 3, 4, 2] becomes [2, 1, -2]). Last entry is ommited
for i in range(len(tmp_daily_temps)):
    if i < len(tmp_daily_temps) - 1:
        daily_temps.append( tmp_daily_temps[i + 1] - tmp_daily_temps[i] )


temps_training = []
temps_labels = []

for i in range(len(daily_temps)):
    if i + num_days_per_sample == len(daily_temps):
        break
    # Daily price differences for n number of days, where n = num_days_per_sample. This is what is provided to the model as input
    temps_training.append(daily_temps[i:i + num_days_per_sample])
    # Price difference for the n+1th day. This is what the model is trying to predict
    temps_labels.append(daily_temps[i + num_days_per_sample])


# Convert everything to Keras tensors, and split the data into training data (to train the model), validation data (to test the model as it is being trained)
# and test data (to test the model after it is fully trained)
x_training = keras.ops.convert_to_tensor(temps_training[:int(len(temps_training) * test_data_split_location)])
y_training = keras.ops.convert_to_tensor(temps_labels[:int(len(temps_labels) * test_data_split_location)])

x_validate = keras.ops.convert_to_tensor(temps_training[int(len(temps_training) * test_data_split_location) : int(len(temps_training) * validation_data_split_location)])
y_validate = keras.ops.convert_to_tensor(temps_labels[int(len(temps_labels) * test_data_split_location) : int(len(temps_labels) * validation_data_split_location)])

x_test = keras.ops.convert_to_tensor(temps_training[int(len(temps_training) * validation_data_split_location) :])
y_test = keras.ops.convert_to_tensor(temps_labels[int(len(temps_labels) * validation_data_split_location) :])

# Create the model
model = keras.Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="mean_squared_error", # loss function for scalar regression
    metrics=["mean_absolute_error"],
)

# Train the model using the training data
history = model.fit(x_training, y_training, 
          validation_data=(x_validate, y_validate),
          batch_size=32, epochs=num_epochs, verbose=1)

# Test the model on the test data
test_loss = model.evaluate(x_test, y_test)
print(f"Test data loss: {test_loss}")


baseline = 0
# Calculate the average error of a model that always predicts no change in temperature from the last temperature change
# This is a baseline to compare the actual model to
for i, val in enumerate([sublist[-1] for sublist in x_test]):
    abs_err = abs(val - y_test[i])

    if baseline == 0:
        baseline = abs_err
    else:
        baseline = (abs_err + baseline) / 2

print(f"Baseline MAE: {baseline}")

# Put the model training statistics into a nice graph (training_graph.png)
mae_history = history.history["val_mean_absolute_error"]

epochs = range(1, len(mae_history) + 1)
plt.plot(epochs, mae_history)
plt.xlabel("Training Iterations")
plt.ylabel("Validation Error Average (°C)")
plt.title("Model Validation Error During Training")
plt.savefig("training_graph.png")
