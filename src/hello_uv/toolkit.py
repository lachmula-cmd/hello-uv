import hello_uv.m0 as m
import hello_uv.imagetools as imagetools
from pathlib import Path
import matplotlib.pyplot as plt
import hello_uv.arrays as arrays



imagess = m.load_images(Path("data") /  "train-images-idx3-ubyte.gz" )
labels = m.load_labels(Path("data") / "train-labels-idx1-ubyte.gz")
images = imagess[0]
inverted = imagetools.invert(images)
double_contrast = imagetools.adjust_contrast(images, 2)
half_contrast = imagetools.adjust_contrast(images, 0.5)
shifted = imagetools.shift(images, 3,2)
rotated = imagetools.rotate(images, 15)

img_list = [inverted,double_contrast,half_contrast,shifted,rotated]



fig, axes = plt.subplots(2,3)
axes[0,0].imshow(inverted)
axes[0,0].set_title("inverted")
axes[0,0].axis("off")

axes[0,1].imshow(double_contrast)
axes[0,1].set_title("contrast 2x")
axes[0,1].axis("off")

axes[0,2].imshow(half_contrast)
axes[0,2].set_title("contrast 0.5")
axes[0,2].axis("off"
               )
axes[1,0].imshow(shifted)
axes[1,0].set_title("shifted")
axes[1,0].axis("off")

axes[1,1].imshow(rotated)
axes[1,1].set_title("rotate 15d")
axes[1,1].axis("off")

axes[1,2].imshow(images)
axes[1,2].set_title("normal")
axes[1,2].axis("off")

for i in img_list:
    cos_sim = arrays.cosine(images, i)
    print(cos_sim)

