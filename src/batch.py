import numpy as np

 # functions for dot products
def flatten_images(images):
    flatten = images.reshape(len(images), -1).astype(float)
    return flatten

def normalise_rows(x):
    length = np.linalg.norm(x, axis=1, keepdims=True)
    return x / length

def cosine_to_all(x,i):
    x_normalized = normalise_rows(x)
    dot_product =  x_normalized @ x_normalized[i]
    return dot_product

def top_k_similar(X, i, k):
    cta = cosine_to_all(X, i)
    positions = np.argsort(cta)[::-1]
    positions = positions[positions != i]
    pos_k = positions[0:k]
    return pos_k

def centre_columns(X):
    col_mean = np.mean(X, axis=0)
    return X - col_mean

