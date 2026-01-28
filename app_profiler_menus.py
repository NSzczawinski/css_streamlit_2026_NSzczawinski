import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title = "Researcher Profile and Distribution Explorer", layout = "wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Current Work", "Distribution Reconstructor", "Contact"],
)

# Sections based on menu selection
if menu == "Researcher Profile":
    st.title("Researcher Profile")
    st.sidebar.header("Profile Options")

    # Collect basic information
    name = "Nicholas Szczawinski"
    field = "Mathematical Statistics"
    institution = "Rhodes University"

    # Display basic profile information
    st.write(f"**Name:** {name}")
    st.write(f"**Field of Research:** {field}")
    st.write(f"**Institution:** {institution}")
    
    st.image(
    "https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg",
    caption="Nature (Pixabay)"
)

elif menu == "Current Work":
    st.title("Current Work")

elif menu == "Distribution Reconstructor":
    st.title("Distribution Reconstructor")
    st.sidebar.header("Distribution Selection")
        
elif menu == "Contact":
    st.header("Contact Information")
    email = "jane.doe@example.com"

    st.write(f"You can reach me at {email}.")





