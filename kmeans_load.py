from utils import kMeans
import numpy as np
from PIL import Image

for k in [4,8,16,32,64,128,256]:
    model = kMeans(k)
    model.codebooks = np.load(f"codebooks_{k}.npy")

    image = Image.open("img.jpeg")
    image = np.array(image)[::2, ::2]/255

    cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
    new_img = np.zeros_like(image)
    for i in range(len(model.codebooks)):
        new_img[cvec==i] = model.codebooks[i]

    Image.fromarray((new_img*255).astype(np.uint8)).save(f"img_{k}.png")