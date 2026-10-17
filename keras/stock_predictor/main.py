# I have always wondered whether AI would be capable of predicting the stock market or crypto price, so I guess this will be a good test

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

import csv

# constants
# data_file_name = "/home/ayles/Programming_Local/python/AIF/AIF-AI-Development/keras/stock_predictor/Bitcoin Historical Data.csv" # daily bitcoin price
data_file_name = "/home/ayles/Programming_Local/python/AIF/AIF-AI-Development/keras/stock_predictor/btc_1h.csv"

num_days_per_sample = 12 # 100 days per sample
num_epochs = 12

test_data_split_location = 0.80 # 80% of the data is training data
validation_data_split_location = 0.90 # 10% of the data is test data

print("Reading data from file")

daily_prices = [] # list to store prices

# Open the CSV file
with open(data_file_name, mode='r') as file:
    reader = csv.DictReader(file)

    # Iterating over each row
    for row in reader:
        daily_prices.append(float(row['open'].replace(",", "")))
        # daily_prices.append(float(row['Price'].replace(",", "")))

# daily_prices.reverse()  # This is for the daily price data

tmp_daily_prices = daily_prices
daily_prices = []

for i in range(len(tmp_daily_prices)):
    if i < len(tmp_daily_prices) - 1:
        daily_prices.append( tmp_daily_prices[i + 1] - tmp_daily_prices[i] )

print(len(daily_prices))

price_training = []
price_labels = []


for i in range(len(daily_prices)):
    if i + num_days_per_sample == len(daily_prices):
        break
    price_training.append(daily_prices[i:i + num_days_per_sample])
    price_labels.append(daily_prices[i + num_days_per_sample])


print(f"Length of training data {len(price_training)}\nLength of labels {len(price_labels)}")

x_training = keras.ops.convert_to_tensor(price_training[:int(len(price_training) * test_data_split_location)])
y_training = keras.ops.convert_to_tensor(price_labels[:int(len(price_labels) * test_data_split_location)])

x_validate = keras.ops.convert_to_tensor(price_training[int(len(price_training) * test_data_split_location) : int(len(price_training) * validation_data_split_location)])
y_validate = keras.ops.convert_to_tensor(price_labels[int(len(price_labels) * test_data_split_location) : int(len(price_labels) * validation_data_split_location)])

x_test = keras.ops.convert_to_tensor(price_training[int(len(price_training) * validation_data_split_location) :])
y_test = keras.ops.convert_to_tensor(price_labels[int(len(price_labels) * validation_data_split_location) :])

mean = keras.ops.mean(x_training) # mean of all prices
std = keras.ops.std(x_training)   # standard deviation of all prices

# normalise the data
x_training = (x_training - mean) / std
y_training = (y_training - mean) / std
x_test = (x_test - mean) / std
y_test = (y_test - mean) / std
x_validate = (x_validate - mean) / std
y_validate = (y_validate - mean) / std

def get_model():
    # function to get a keras model
    model = keras.Sequential(
        [
            # layers.Dense(128, activation="relu"),
            layers.Dense(32, activation="relu"),
            # layers.Dense(4, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error", # loss function for scalar regression
        metrics=["mean_absolute_error"],
    )
    return model

model = get_model()

history = model.fit(x_training, y_training, 
          validation_data=(x_validate, y_validate),
          batch_size=32, epochs=num_epochs, verbose=1)

test_loss = model.evaluate(x_test, y_test)
print(f"Test data loss: {test_loss}")

mae_history = history.history["val_mean_absolute_error"]

epochs = range(1, len(mae_history) + 1)
plt.plot(epochs, mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.savefig("/home/ayles/Programming_Local/python/AIF/AIF-AI-Development/keras/stock_predictor/training_graph.png")

