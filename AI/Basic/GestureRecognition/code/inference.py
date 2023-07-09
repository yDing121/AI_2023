import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

model = tf.keras.models.load_model("./test_model")
thing = pd.read_csv("test.csv").to_numpy()
print(thing)
print(model.predict(thing))