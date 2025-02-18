import streamlit as st
import numpy as np
from PIL import Image

class kMeans():
    def __init__(self, num_cluster=5, data=None):
        self.codebooks = 3*np.random.rand(num_cluster, 2)-1.5

    def pred(self, x):
        dists = np.sqrt(np.sum((x[:,None]-self.codebooks[None])**2, axis=-1)).T
        return np.argmin(dists, axis=0)

    def fehler(self, x):
        dists = np.sqrt(np.sum((x[:,None]-self.codebooks[None])**2, axis=-1)).T
        d = np.argmin(dists, axis=0)
        out = 0
        for j,i in enumerate(d):
            out += dists[i,j]
        return out/len(x)

st.title("Farbreduktion durch Clustering")

image = Image.open("img.jpeg")
image = np.array(image)[::2, ::2]/255
st.image(image, caption="Originalbild", use_container_width=True)

# Wähle einen Wert für, Auswahlmöglichkeiten 4, 8, 16, 32
k = st.selectbox("Wähle Anzahl der Farben", [4, 8, 16, 32])
model = kMeans(k)
model.fit(image.reshape(-1, 3), verbose=False, max_it=100)
cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
#print(cvec)
new_img = np.zeros_like(image)
for i in range(k):
    new_img[cvec==i] = model.codebooks[i]

st.image(new_img, caption=f"kMeans-Annäherung, Fehler {model.fehler(image.reshape(-1,3)):.4f}", use_container_width=True)
