import streamlit as st
import numpy as np
from PIL import Image
from utils import kMeans

st.title("Farbreduktion durch Clustering")

image = Image.open("img.jpeg")
image = np.array(image)[::2, ::2]/255
st.image(image, caption="Originalbild", use_container_width=True)

# Platzhalter für Colorpicker
st.subheader("Wähle 8 Codebookvektoren")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
selected_colors = []
with col1:
    color = st.color_picker("Farbe 1", "#000000")
    selected_colors.append(color)
with col2:
    color = st.color_picker("Farbe 2", "#000000")
    selected_colors.append(color)
with col3:
    color = st.color_picker("Farbe 3", "#000000")
    selected_colors.append(color)
with col4:
    color = st.color_picker("Farbe 4", "#000000")
    selected_colors.append(color)
with col5:
    color = st.color_picker("Farbe 5", "#000000")
    selected_colors.append(color)
with col6:
    color = st.color_picker("Farbe 6", "#000000")
    selected_colors.append(color)
with col7:
    color = st.color_picker("Farbe 7", "#000000")
    selected_colors.append(color)
with col8:
    color = st.color_picker("Farbe 8", "#000000")
    selected_colors.append(color)
    
#for c in selected_colors:
st.write(", ".join(selected_colors))

model = kMeans(8)
model.codebooks = np.array([np.array([int(c[1:3]/255, 16), int(c[3:5]/255, 16), int(c[5:]/255, 16)]) for c in selected_colors])
cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
#print(cvec)
new_img = np.zeros_like(image)
for i in range(8):
    new_img[cvec==i] = model.codebooks[i]

st.image(new_img, caption=f"kMeans-Annäherung, Fehler {model.fehler(image.reshape(-1,3)):.4f}", use_container_width=True)

if False:
    model.fit(image.reshape(-1, 3), verbose=True)
    cvec = model.pred(image.reshape(-1, 3)).reshape(image.shape[:2])
    print(cvec)
    new_img = np.zeros_like(image)
    for i in range(8):
        new_img[cvec==i] = model.codebooks[i]

    st.image(new_img, caption="kMeans-Annäherung", use_container_width=True)