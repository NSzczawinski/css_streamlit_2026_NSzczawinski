import streamlit as st
import pandas as pd
import time
from PIL import Image
import os

# Set page title
st.set_page_config(page_title = "Nicholas Szczawinski Streamlit Profile", layout = "wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Current Work", "Distribution Reconstructor", "Contact"],
)

# Sections based on menu selection
if menu == "Researcher Profile":
    # Title and heading
    st.title("Researcher Profile")
    st.header("Nicholas Szczawinski")
    st.write("Master's Student in Mathematical Statistics - Researching Bivariate Distribution Reconstruction via Neural Networks")
    st.divider()
    
    # Column initialisation
    column1, column2 = st.columns(2)
    
    # Left column content
    with column1:
        st.subheader("Education")
        st.write("""
        **MSc in Mathematical Statistics (Rhodes University : 2026)**\n
        **Thesis:** Building a Multivariate Neural Network Sampler
        """)
        st.divider()
        st.write("""
        **BScH in Mathematical Statistics *With Distinction* (Rhodes University : 2025)**\n
        **Research Project:** Building a Feed-forward Characteristic Function Matching Generator\n
        **Awards:**\n
        - Dean's List for Academic Merit
        - Rhodes University Postgraduate Scholarship
        - Rhodes University Foundation Scholarship
        """)
        st.divider()
        st.write("""
        **BSc (Rhodes University: 2022-2024) Triple Major:**\n
          **Applied Mathematics** *With Distinction*\n
          **Economics** *With Distinction*\n
          **Mathematical Statistics** *With Distinction*\n
        **Awards:**\n
        - Dean's List for Academic Merit
        - RL Threlfell Memorial Prize for best final year Economics student (2024)
        - 1st place for Mathematical Statistics (2024)
        - 2nd place for Mathematical Statistics (2023)
        - Joint 5th place for Economics (2023)
        - Top 10 Achievers List for Economics (2022)
        - GRPGV Academic Award (2022, 2024)
        - GRPGV Leadership Award (2022, 2024)
        """)
        st.divider()
        st.write("""
        **BSocScH in Organisational Psychology *With Distinction* (Rhodes University : 2021)**\n
        **Research Project:** Organisational Cognitive Neuroscience and the Relevance of a Biologically-Based Interpretation of Organisational Phenomena\n
        **Awards:**\n
        - Rhodes University Postgraduate Scholarship
        - Academic Colours
        """)
        st.divider()
        st.write("""
        **BSocSc (Rhodes University: 2022-2024) Double Major:**\n
          **Organisational Psychology** *With Distinction*\n
          **Industrial & Economic Sociology** *With Distinction*\n
        **Awards:**\n
        - Dean's List for Academic Merit
        - Kimberley Hall Academic Award (2020)
        - ABSA Bank Scholarship (2018, 2019, 2020)
        """)
        st.divider()
        
    # Right column content
    with column2:
        st.subheader("Research Interests")
        st.write("""
        - Statistical Learning and Neural Networks
        - Probability Theory
        - Numerical Methods for Sampling
        """)
        st.subheader("Details")
        st.write("""
        - Location: Makhanda/Grahamstown - Eastern Cape - South Africa 
        - Programming Languages: R, Python, MATLAB, Java
        - Contact: www.linkedin.com/in/nickszczawinski
        """)

elif menu == "Current Work":
    st.title("Current Work")

elif menu == "Distribution Reconstructor":
    st.title("Distribution Reconstructor")
    # -------------------------------------------------------------------------------------------------------------
    # Loss CSV
    df = pd.read_csv("streamlit_eg_1/loss_vector (mod 100).csv", header = None)
    df.columns = ["Loss value"]
    df.index.name = "Pass mod 100"
    
    st.header("Equally Weighted Bivariate Normal Mixture Distribution - Demo")
    st.write("This is an example of how my neural network reconstructs a distribution by learning the characteristic function over 10 000 passes")
    
    # Start button
    if "running" not in st.session_state:
        st.session_state.running = False
    if st.button("Start Demo"):
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
    
            time.sleep(0.5)
    # -------------------------------------------------------------------------------------------------------------
        
elif menu == "Contact":
    st.header("Contact Information")
    email = "jane.doe@example.com"
    st.write(f"You can reach me at {email}.")









