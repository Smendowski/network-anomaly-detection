import time
from statistics import mean

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import  L1L2
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

tf.config.list_physical_devices('GPU')

t = time.time()
df = pd.read_csv("final_dataset.csv")
print(time.time() - t)

import numpy as np
t = time.time()

# first 4 columns contain IPs and Ports
data = df.values[:, 5:]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=21)

train_data = np.asarray(train_data).astype('float32')
test_data = np.asarray(test_data).astype('float32')
print(time.time() - t)

class AnomalyDetector(Model):
  def __init__(self, i):
    print(i)
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(86, activation="relu"),
      layers.Dense(43, activation="relu"),
      layers.Dense(30, activation="relu"),
      layers.Dense(20, activation="relu"),
      layers.Dense(i, activation="linear", activity_regularizer=L1L2(0.00001))])

    self.decoder = tf.keras.Sequential([
      layers.Dense(20, activation="relu"),
      layers.Dense(30, activation="relu"),
      #layers.Dense(86, activation="relu"),
      layers.Dense(43, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

for i in range(8,9):
    autoencoder = AnomalyDetector(i)

    autoencoder.compile(optimizer='adam', loss='mae')

    history = autoencoder.fit(train_data, train_data,
              epochs=10,
              batch_size=512,
              validation_data=(test_data, test_data),
              shuffle=True)

    t = time.time()
    reconstruction = autoencoder.predict(test_data)
    t1 = time.time() - t

    print(t1)
    print(t1/len(test_data))
    autoencoder.save("D:\Projekty\ML")
    print("Model saved")
    errMatrix = abs(test_data-reconstruction)
    errList = []

    # MAE
    for i in range(0, len(errMatrix[1,:])):
        errList.append(mean(errMatrix[:, i]))

    # MSE
    #for i in range(0, len(errMatrix[1,:])):
    #    x = np.multiply(errMatrix[:, i], errMatrix[:, i])
    #    errList.append(mean(x))

    err = mean(errList)
    print(err)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


df.values[1,5:]
