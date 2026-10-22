from m0 import load_labels
from m0 import load_images
import arrays
from pathlib import Path
import matplotlib.pyplot as plt

labels = load_labels(Path("data") / "train-labels-idx1-ubyte.gz")
images = load_images(Path("data") / "train-images-idx3-ubyte.gz")

print(f"center : {arrays.crop_center(images[0], 14)}")
print(f"count_bright:{arrays.count_bright(images[0], 200)}")
print(f"row_mean = {round(arrays.row_means(images[0])[14], 2)}")
print(f"cosine-sim: {labels[11]}, {round(arrays.cosine(images[0], images[11]),3)}")
print(f"{round(arrays.cosine(images[0], images[1]), 3)}")


plt.imshow(arrays.flip_horizontal(images[0]), cmap = "gray")
plt.title(labels[0])
plt.savefig("-1.png")
plt.close()