import gzip
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def load_labels(path):
    with gzip.open(path, "rb") as f:                               # open the .gz for reading bytes
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)   # skip the 8-byte header

labels = load_labels(Path("data") / "train-labels-idx1-ubyte.gz")

def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        
        return data.reshape(len(labels),28,28)
    
images = load_images(Path("data") / "train-images-idx3-ubyte.gz" )
    
if __name__ == "__main__":
    
    print(labels.shape, labels[4], CLASSES[labels[4]])
    
    print(CLASSES[labels[4]])
    print(images.shape)

    v0 = images[0].reshape(-1).astype(float)
    print(v0.shape, np.linalg.norm(v0))

    v1 = images[1].reshape(-1).astype(float)
    print(v1.shape, np.linalg.norm(v1))