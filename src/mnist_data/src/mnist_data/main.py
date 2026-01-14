import numpy as np

from sklearn.datasets import load_digits

ds = load_digits()

X = ds.data.astype(np.float32) / 16.0 # normalised input as tensor
y = ds.target.astype(np.uint8) # unsigned integer, only from 0-9

np.save("../data/digits_X.npy", X)
np.save("../data/digits_y.npy", y)