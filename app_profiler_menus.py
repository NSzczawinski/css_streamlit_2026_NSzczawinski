import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title = "Researcher Profile and Distribution Explorer", layout = "wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Publications", "Distribution Explorer", "Contact"],
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

elif menu == "Publications":
    st.title("Publications")
    st.sidebar.header("Upload and Filter")

    # Upload publications file
    uploaded_file = st.file_uploader("Upload a CSV of Publications", type="csv")
    if uploaded_file:
        publications = pd.read_csv(uploaded_file)
        st.dataframe(publications)

        # Add filtering for year or keyword
        keyword = st.text_input("Filter by keyword", "")
        if keyword:
            filtered = publications[
                publications.apply(lambda row: keyword.lower() in row.astype(str).str.lower().values, axis=1)
            ]
            st.write(f"Filtered Results for '{keyword}':")
            st.dataframe(filtered)
        else:
            st.write("Showing all publications")

        # Publication trends
        if "Year" in publications.columns:
            st.subheader("Publication Trends")
            year_counts = publications["Year"].value_counts().sort_index()
            st.bar_chart(year_counts)
        else:
            st.write("The CSV does not have a 'Year' column to visualize trends.")

elif menu == "Distribution Explorer":
    st.title("Distributions")
    st.sidebar.header("Distribution Selection")
    
    # Tabbed view for STEM data
    data_option = st.sidebar.selectbox(
        "Choose a distribution", 
        ["Normal", "Gamma", "Normal Inverse Gaussian"]
    )

    if data_option == "Normal":
        st.write("### Normal Distribution")

    elif data_option == "Gamma":
        st.write("### Gamma Distribution")

    elif data_option == "Normal Inverse Gaussian":
        st.write("### Normal Inverse Gaussian Distribution")
        
elif menu == "Contact":
    # Add a contact section
    st.header("Contact Information")
    email = "jane.doe@example.com"

    st.write(f"You can reach me at {email}.")




