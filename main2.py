import streamlit as st
import numpy as np
from PIL import Image
from utils import kMeans

st.title("Farbreduktion durch Clustering")

image = Image.open("img.jpeg")
image = np.array(image)[::2, ::2]/255
st.image(image, caption="Originalbild", use_container_width=True)

# Wähle einen Wert für, Auswahlmöglichkeiten 4, 8, 16, 32
k = st.slider("Wähle die Anzahl der Farben", 4, 32, 8, 4)
model = kMeans(k)
model.fit(image.reshape(-1, 3), verbose=False, max_it=100)
cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
#print(cvec)
new_img = np.zeros_like(image)
for i in range(8):
    new_img[cvec==i] = model.codebooks[i]

st.image(new_img, caption=f"kMeans-Annäherung, Fehler {model.fehler(image.reshape(-1,3)):.4f}", use_container_width=True)
