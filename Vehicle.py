import streamlit as st
import numpy as np
import json
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Vehicle Classification", page_icon=":bus:", layout="wide")

def get(path:str):
    with open(path,"r") as p:
        return json.load(p)


robot_path = get("./assets/chuchu.json")
team_path = get("./assets/team.json")
ano_path = get("./assets/Ano.json")
to_path = get("./assets/to.json")



# Sidebar
with st.sidebar:
    selected = option_menu(
         menu_title = "Main Menu",
        options = ["Home", "About Project", "Vehicle Classification", "Team"],
        icons = ["house", "book", "pin","people"],
        menu_icon ="cast",
        default_index = 0,
    )
       

# Home Page
if selected == "Home":
    st.header("Hi, We are Group 2 :wave:")
    st.write("")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
            st.write("<div style ='text-align: center;'> Vehicle Classification </div>, unsafe_allow_html=True)
        with left_column:
            st.write(
                """
                <div style='text-align: justify;'>
                    Vehicle classification is crucial for intelligent transportation systems, focusing on identifying and categorizing buses, trucks, cars, and motorcycles. 
                    Buses facilitate public transit, trucks handle large-scale freight, cars provide personal travel flexibility, and motorcycles offer efficient short-distance travel.
                    Accurate classification enhances traffic management, law enforcement, and road safety. Understanding the specific roles and characteristics of each vehicle type enables
                    more effective traffic control, regulatory compliance, and targeted safety measures. This leads to a more efficient and safer transportation network, benefiting both authorities
                    and road users. 
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_column:
            st_lottie(robot_path, height = 250, key = "hi")

# ABOUT PROJECT
if selected == "About Project":
    st.header("About Project")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(
                """
                <div style='text-align: justify;'>
                    Vehicle classification is crucial for intelligent transportation systems, focusing on identifying and categorizing buses, trucks, cars, and motorcycles. 
                    Buses facilitate public transit, trucks handle large-scale freight, cars provide personal travel flexibility, and motorcycles offer efficient short-distance travel.
                    Accurate classification enhances traffic management, law enforcement, and road safety. Understanding the specific roles and characteristics of each vehicle type enables
                    more effective traffic control, regulatory compliance, and targeted safety measures. This leads to a more efficient and safer transportation network, benefiting both authorities
                    and road users. 
                </div>
                """,
                unsafe_allow_html=True
            )
        with right_column:
            st_lottie(ano_path, height = 250, key = "hi")

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with right_column:
            st.header("DATASET")
            st.write(
                """
                <div style='text-align: justify;'>
                    The dataset for this project came from this link: <a href='https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition' target='_blank'>Kaggle Vehicle Type Recognition</a>. <br>
                    It contains the following: <br>
                    - TRUCK  <br>
                    - BUS    <br>
                    - CAR    <br>
                    - MOTORCYCLE  <br>
                </div>
                """,
                unsafe_allow_html=True
           )
        with left_column:
                st_lottie(to_path, height=250, key="h1")


# Vehicle Classification
if selected == "Vehicle Classification":
    st.header("Vehicle Classification")
    
    with st.container():
        uploaded_image = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
            
            try:
                image = Image.open(uploaded_image)
                image = image.resize((224, 224))
                image = np.expand_dims(image, axis=0)
                image = np.array(image)

                pred_probabilities = model.predict(image)
                pred_class_index = np.argmax(pred_probabilities, axis=1)[0]

                if pred_class_index in vehicle_names:
                    predicted_vehicle = vehicle_names[pred_class_index]
                    st.success(f"Prediction: {predicted_vehicle}")
                else:
                    st.warning("Unknown Vehicle")
            except Exception as e:
                st.warning("Error processing image. Please upload a valid image.")


# Team Page
if selected == "Team":
    st.header("Meet the Team")
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.write(
                """
                Dela cruz, Ivan Kenneth B.<br>
                Empaynado, Shenia J.<br>
                Españo, Rens S.<br>
                Eustaquio, Neil Marco C.<br>
                Fermin, Jozette P.<br>
            </div>
            """,
                unsafe_allow_html=True
        )
        with right_column:
            st_lottie(team_path, height = 150, key = "hii")
