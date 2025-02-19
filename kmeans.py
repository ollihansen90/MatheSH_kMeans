from utils import kMeans
from PIL import Image
import numpy as np

image = Image.open("img.jpeg")
image = np.array(image)[::2, ::2]/255

k = 256
model = kMeans(k)
model.fit(image.reshape(-1, 3), verbose=True, max_it=100)
np.save(f"codebooks_{k}.npy", model.codebooks)