import streamlit as st
import numpy as np
from PIL import Image
import time

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
k = st.selectbox("Wähle Anzahl der Farben", [4, 8, 16, 32, 64, 128, 256])
fehlerdict = {4: 0.060077662393129276, 8: 0.0393518538412903, 16: 0.026051415851631534, 32: 0.015555190300325317, 64: 0.011502079846342567, 128: 0.007934572557017135, 256: 0.005912390338907347}
new_img = Image.open(f"img_{k}.png")
new_img = np.array(new_img)[::2, ::2]/255

bar = st.progress(0, text=f"Lade kMeans mit {k} Zentren... Bitte warten...")

for percent_complete in range(100):
    time.sleep(0.1*k/256)
    bar.progress(percent_complete + 1, text=f"Lade kMeans mit {k} Zentren... Bitte warten...")
bar.empty()
st.image(new_img, caption=f"kMeans-Annäherung, Fehler {fehlerdict[k]:.4f}", use_container_width=True)
