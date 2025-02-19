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

# Platzhalter für Colorpicker
st.subheader("Wähle 8 Codebookvektoren")
col1, col2, col3 = st.columns([1,1,6])
selected_colors = []
with col1:
    for i in range(4):
        color = st.color_picker(f"Farbe {i+1}", "#000000")
        selected_colors.append(color)
with col2:
    for i in range(4,8):
        color = st.color_picker(f"Farbe {i+1}", "#000000")
        selected_colors.append(color)
with col3:
    st.image(image, caption="Originalbild", use_container_width=True)

    
#selected_colors = st.text_input(value=", ".join(selected_colors)).split(", ")

selected_colors = st.text_input(label="Faben als HEX", value=", ".join(selected_colors)).split(", ")
model = kMeans(8)
model.codebooks = np.array([np.array([int(c[1:3], 16)/255, int(c[3:5], 16)/255, int(c[5:], 16)/255]) for c in selected_colors])
cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
#print(cvec)
new_img = np.zeros_like(image)
for i in range(8):
    new_img[cvec==i] = model.codebooks[i]

st.image(new_img, caption=f"kMeans-Annäherung, Fehler {model.fehler(image.reshape(-1,3)):.4f}", use_container_width=True)
