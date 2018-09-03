
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import numpy as np
import os
import matplotlib.pyplot as plt

#dataset
scaler = StandardScaler()
#-housing_data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m ,1)),housing.data]
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

#method
def reset_graph(seed = 42):
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  np.random.seed(seed)


def fetch_batch(epoch, batch_index, batch_size, n_batchs):
  np.random.seed(epoch * n_batchs + batch_index)
  indices = np.random.randint(m, size = batch_size)
  X_batch = scaled_housing_data_plus_bias[indices]
  y_batch = housing.target.reshape(-1, 1)[indices]
  return X_batch, y_batch

PROJECT_ROOT_DIR = "/Users/ryousuke/desktop/ml/tensorflow_practice/2_dnn"
CHAPTER_ID = "deep"

def save_fig(fig_id, tight_layout=True):
  path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
  print("Saving figure", fig_id)
  if tight_layout:
    plt.tight_layout()
  plt.savefig(path, format = "png", dpi = 300)

def logit(z):
  return  1/(1+np.exp(-z))
