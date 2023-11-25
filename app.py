import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist
import os

@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs , all_names
vecs , names = read_data()

st.title("Product Recommender")
st.write("Step1: Select a random image ")
ch = st.button("Random Image")
if ch:
    random_name = names[np.random.randint(len(names))]
    st.image(Image.open("./images/" + random_name) , width=200)
    st.session_state["disp_img"] = random_name

st.write("Step2: Click on this button to recommend the products")
fs = st.button("Recommend Images")
if fs:
    st.write("Selected Image")
    st.image(Image.open("./images/" + st.session_state["disp_img"]) , width = 140)
    st.write("5 Recommended Images")
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]
    top5 = cdist(target_vec[None , ...] , vecs).squeeze().argsort()[1:6]
    c1.image(Image.open("./images/" + names[top5[0]]), width = 140)
    c2.image(Image.open("./images/" + names[top5[1]]), width = 140)
    c3.image(Image.open("./images/" + names[top5[2]]), width = 140)
    c4.image(Image.open("./images/" + names[top5[3]]), width = 140)
    c5.image(Image.open("./images/" + names[top5[4]]), width = 140)