import streamlit as st


# Set page title
st.set_page_config(page_title = "Researcher Profile and Distribution Reconstructor", layout = "wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Current Work", "Distribution Reconstructor", "Contact"],
)

# Sections based on menu selection
if menu == "Researcher Profile":
    st.title("Researcher Profile")

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
    
    random_seed= st.checkbox("Use a random seed?", value = False)
    samples_count = st.number_input("How many samples do you want to reconstruct?:", value = 5000)
    frequencies_count = st.number_input("How many samples of the characteristic function do you want to take?:", value = 100)
    
    topology_options = [5000, 2500, 1000, 500, 300, 100, 50, 25, 10, 2]
    topology_list = st.multiselect("Select neuron layers in order (must end with 2):", topology_options, default = [300, 50, 2])
    st.write("Your list:", topology_list)

    activation_options = ["sigmoid", "ReLU", "purelin", "sin", "tanh"]
    activation_list = st.multiselect("Select activation functions in order:", activation_options, default = ["sin", "sin", "purelin"])
    st.write("Your list:", activation_list)
    
    selected_loss = st.session_state.get("selected_loss", [])
    def update_selection():
        st.session_state.selected = selected_loss
    activation_options = ["SEL", "SEL+"]
    max_choices = 1
    if len(activation_list) > max_choices:
        st.warning(f"You can select at most {max_choices} loss function")

    batchsize = st.text_input("Select batchsize:", value = "fullbatch")
    resample_factor = st.number_input("After how many iterations do you want to resample the noise?:", value = 100)
    
    lr_options = ["0.1", "0.01", "0.001", "0.0001", "0.00001"]
    lr_list = st.multiselect("Select a learn rate:", lr_options, default = ["0.001"])
    
    passes = st.number_input("How many iterations should the network complete?:", value = 10000)

    loss_options = ["SEL", "SEL+"]
    loss_list = st.multiselect("Select the loss type:", loss_options, default = "SEL+")

#if st.button("Run NN"):
    # Call the NN function
    # result = NN(user_input)
    # st.success(f"The NN output is: {result}")
        
elif menu == "Contact":
    st.header("Contact Information")
    email = "jane.doe@example.com"

    st.write(f"You can reach me at {email}.")
















