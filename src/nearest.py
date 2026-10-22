import batch
import m0
from pathlib import Path 
from arrays import cosine

images = m0.load_images(Path("data") / "train-images-idx3-ubyte.gz")
labels = m0.load_labels(Path("data") / "train-labels-idx1-ubyte.gz")

image_1000 = images[:1000]

x = batch.flatten_images(image_1000)
print(f"flatted size : {x.shape}")
print(f"cosine : {round(cosine(images[0], images[11]), 3)}")
print(f"cosine : {round(cosine(images[0], images[1]), 3)}")
print(f"cosine : {round(cosine(images[0], images[0]), 3)}")
similarity = batch.top_k_similar(x, 0, 5)
print(similarity)
print(labels[similarity])