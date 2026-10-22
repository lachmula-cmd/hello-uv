import numpy as np

def crop_center(img, size):
 print(img.shape)
 r_length = img.shape[0]
 c_length = img.shape[1]
 r_cnt = (r_length - size) // 2
 c_cnt = (c_length - size) // 2
 return img[r_cnt: r_cnt + size, c_cnt: c_cnt + size]

def flip_horizontal(img):
    return img[:,::-1]

def count_bright(img, threshold):
   count = int((img > threshold).sum())
   return count

def row_means(img):
   return img.mean(axis=1)

def normalise(img):
   img_fl = img.reshape(-1).astype(float)
   return img_fl / 255

def cosine(a,b):
   a1 = a.reshape(-1).astype(float)
   b1 = b.reshape(-1).astype(float)
   dot_ab = np.dot(a1,b1)
   cos_sim = dot_ab / (np.linalg.norm(a1) * np.linalg.norm(b1))
   return cos_sim.astype(float) 





