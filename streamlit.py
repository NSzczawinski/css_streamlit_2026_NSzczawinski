import streamlit as st
import pandas as pd
import time
from PIL import Image
import os

# Loss CSV
df = pd.read_csv("loss_vector (mod 100).csv", header = None)
df.columns = ["Loss value"]
df.index.name = "Pass mod 100"

st.header("Equally Weighted Bivariate Normal Mixture Distribution")

if "running" not in st.session_state:
    st.session_state.running = False
if st.button("Start Demo (10 000 passes)"):
    st.session_state.running = True

# Placeholders for line graphs and plots
col1, col2 = st.columns(2)
CF_image_placeholder = col1.empty()
Density_image_placeholder = col2.empty()
chart = st.line_chart()

# Folders with images
CF_image_folder = "CF_images"
Density_image_folder = "Density_images"

# Yes I know the following code sucks, its because my programme cant produce plots when no passes have been run. So my plots start at 1 pass, not 0.
if st.session_state.running:
    for i in range(1, len(df) + 1):
        chart.add_rows(df.iloc[i:i+1])
        if i == 1:
            factor = 1
            ith_CF_image = f"CF_plots_{i * factor}.png"
            CF_image_path = os.path.join("streamlit_eg_1/CF_images", ith_CF_image)
            CF_image = Image.open(CF_image_path)
            CF_image_placeholder.image(CF_image, caption = f"Characteristic Function at Pass {i * factor}")
            
            ith_Density_image = f"Density_plot_{i * factor}.png"
            Density_image_path = os.path.join("streamlit_eg_1/Density_images", ith_Density_image)
            Density_image = Image.open(Density_image_path)
            Density_image_placeholder.image(Density_image, caption = f"Density Function at Pass {i * factor}")
        elif i % 10 == 0:
            factor = 100
            ith_CF_image = f"CF_plots_{i * factor}.png"
            CF_image_path = os.path.join("streamlit_eg_1/CF_images", ith_CF_image)
            CF_image = Image.open(CF_image_path)
            CF_image_placeholder.image(CF_image, caption = f"Characteristic Function at Pass {i * factor}")
            
            ith_Density_image = f"Density_plot_{i * factor}.png"
            Density_image_path = os.path.join("streamlit_eg_1/Density_images", ith_Density_image)
            Density_image = Image.open(Density_image_path)
            Density_image_placeholder.image(Density_image, caption = f"Density Function at Pass {i * factor}")
        time.sleep(0.1)



