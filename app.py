import streamlit as st
import numpy as np
from PIL import Image
from data_processing import process_local_data, process_image
from model_analysis import prepare_model_inputs, split_dataset, build_train_ridge_model

def predict(uploaded_file, genre_selected):
    if uploaded_file is not None and genre_selected is not None:
        with st.spinner("Wait for it...", show_time=False):
            image_data, genre_data, all_genres, rating_data = process_local_data()
            x = prepare_model_inputs(image_data, genre_data)
            x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, rating_data)
            trained_ridge_model = build_train_ridge_model(x_train, y_train)

            image_array = process_image(Image.open(uploaded_file))
            genre_array = [1 if genre == genre_selected else 0 for genre in all_genres]
            x_inputs = prepare_model_inputs(np.array([image_array]), np.array([genre_array]))
            y_predict = trained_ridge_model.predict(x_inputs)
            st.write("Predicted IMDB Rating: " + str(round(y_predict.item(0), 2)))

def main():
    st.write("""
    # 🎬 CineGauge - Movie Poster Analyzer

    ## 1. Upload your movie poster
    """)
    uploaded_file = st.file_uploader("Choose a file (jpg, jpeg, png)", type=["jpg","jpeg","png"])
    

    st.write("""
    ## 2. Choose your target genre
    """)
    genre_selected = st.selectbox(
        "Select one genre",
        ("Drama", "Comedy", "Romance", "Science Fiction", "Adventure", "Action", "Biography", "History", "Sport", "Thriller", "Horror", "War", "Fantasy", "Crime", "Mystery", "Animation", "Musical", "Documentary", "Family", "Western", "Film-Noir"),
    )

    st.write("""
    ## 3. Predict the rating of the movie
    """)

    st.button("Analyze", on_click=predict(uploaded_file, genre_selected), disabled=(uploaded_file is None or genre_selected is None))

if __name__ == "__main__":
    main()
