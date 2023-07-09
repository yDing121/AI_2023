import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import os


# Class -> one hot
def is_gesture(ohcode, tag):
    if ohcode == tag:
        return 1
    else:
        return 0


# Load data
x = pd.read_csv("train_stable.csv")
y = pd.read_csv("targets_stable.csv")
print(x.shape)
print(y.shape)

# Class -> one hot
y["rock"] = y.apply(lambda row: is_gesture(row["cat"], 0), axis=1)
y["paper"] = y.apply(lambda row: is_gesture(row["cat"], 1), axis=1)
y["scissors"] = y.apply(lambda row: is_gesture(row["cat"], 2), axis=1)
y = y.drop("cat", axis=1)

together = x.merge(y, on="fname", how="left").drop("fname", axis=1)
print(together.head(5))

y = together[["rock", "paper", "scissors"]].to_numpy(dtype=int)
x = together.drop(["rock", "paper", "scissors"], axis=1).to_numpy(dtype=float)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=69)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

tf.random.set_seed(69)

# Model definition
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=50, activation="relu", input_shape=(42,)),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=50, activation="relu"),
        # tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=3, activation="softmax")
    ]
)

# Compile Model
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callback
lr_sched = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# Training
history = model.fit(
    x,
    y,
    epochs=20,
    # callbacks=lr_sched
)

# Evaluation
model.evaluate(x_test, y_test)

# Graph
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, loss, 'r+-', label='Training Loss')
plt.title('Training Accuracy and Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('history.png')
plt.show()

# Export
plot_model(model, show_shapes=True)
path = "./test_model"
model.save(path)
