import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import sqlite3
import bcrypt
from PIL import Image

# Database connection
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# Create users table
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
''')
conn.commit()

# Password hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Verify password
def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

# Register user
def register_user(username, password):
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

# Login user
def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result and check_password(password, result[0]):
        return True
    return False

# Model prediction function
def model_prediction(image_file):
    model = tf.keras.models.load_model("my_model.keras")
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Sidebar
st.sidebar.title("🔐 Authentication & Navigation")
menu = st.sidebar.selectbox("Select Page", ["HOME", "Disease recognition", "AI chatbot", "About", "Login", "Register", "Logout"])

# Show logged-in user in sidebar
if st.session_state.logged_in:
    st.sidebar.success(f"✅ Logged in as: {st.session_state.username}")


# tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("my_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])#convert sigle img into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index



#home page 
if(menu=="HOME"):
    st.header("AI BASED SOLUTION FOR CROPS DISEASE DETECTION")
    img_path = "/Users/mahesh/Desktop/desktop/AI based sloution for crops/logo/main.jpg"
    st.image(img_path,use_column_width=True)
    st.markdown("""
    Welcome to the AI BASED SOLUTION FOR CROPS DISEASE  DETECTION! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif(menu=="About"):
    st.header("About ")
    st.markdown(""" 
                #### About DATASET
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
   ##### content
    1.Train(70295) images
    2. vaild(17572)images
    3. test(33 images)    """) 



elif menu == "Disease recognition":
    st.header("🌿 Disease Recognition")
    test_image = st.file_uploader("Upload an image of a plant leaf:")

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("🔍 Analyzing Image...")
            result_index = model_prediction(test_image)

            # Class labels
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            predicted_class = class_names[result_index]
            st.success(f"🌱 Predicted Disease: **{predicted_class}**")


# elif menu=="Ai chatbot":


#     # Define the prompt template
#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content="You are a helpful AI assistant."),
#         MessagesPlaceholder(variable_name="context"),
#         HumanMessage(content="{question}")
#     ])

#     # Initialize the model
#     model = OllamaLLM(model="llama3")

#     # Create a chain
#     chain = prompt | model

#     # Streamlit UI
#     st.header("AI Chatbot for Fertilizer Recommendation 🌿🔍")

#     st.markdown("""
#     Copy the disease name and paste it in the search box. 
#     Our AI chatbot will provide good fertilizers to improve crop growth!
#     """)

#     # Display image
#     img_path = "/Users/mahesh/Desktop/desktop/AI based sloution for crops/logo/logo_4.webp"
#     st.image(img_path, use_column_width=True)

#     # Initialize session state for chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat history
#     for role, content in st.session_state.messages:
#         if role == "user":
#             st.chat_message("user").write(content)
#         else:
#             st.chat_message("assistant").write(content)

#     # User input
#     user_input = st.chat_input("Type your query here...")

#     if user_input:
#         # Append user message
#         st.session_state.messages.append(("user", user_input))
#         st.chat_message("user").write(user_input)

#         # Get AI response
#         result = chain.invoke({"context": st.session_state.messages, "question": user_input})

#         # Append AI response
#         st.session_state.messages.append(("assistant", result))
#         st.chat_message("assistant").write(result)

elif menu == "AI chatbot":

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="context"),
        HumanMessage(content="{question}")
    ])

    # Initialize the model
    model = OllamaLLM(model="llama3")

    # Create a chain
    chain = prompt | model

    # Streamlit UI
    st.header("AI Chatbot for Fertilizer Recommendation 🌿🔍")

    st.markdown("""
    Copy the disease name and paste it in the search box. 
    Our AI chatbot will provide good fertilizers to improve crop growth!
    """)

    # Display image
    img_path = "/Users/mahesh/Desktop/desktop/AI based sloution for crops/logo/logo_4.webp"
    st.image(img_path, use_column_width=True)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role, content = message
        st.chat_message(role).write(content)

    # User input
    user_input = st.chat_input("Type your query here...")

    if user_input:
        # Append user message
        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").write(user_input)

        # Get AI response
        result = chain.invoke({"context": st.session_state.messages, "question": user_input})

        # Append AI response and display it
        st.session_state.messages.append(("assistant", result))
        st.chat_message("assistant").write(result)

# LOGIN PAGE
elif menu == "Login":
    st.header("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"🎉 Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid credentials.")

# REGISTER PAGE
elif menu == "Register":
    st.header("📝 Register")
    new_username = st.text_input("Create a Username")
    new_password = st.text_input("Create a Password", type="password")

    if st.button("Register"):
        if register_user(new_username, new_password):
            st.success("✅ Account created successfully!")
        else:
            st.error("⚠️ Username already exists.")

# LOGOUT PAGE
elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("✅ You have been logged out.")
    st.experimental_rerun()