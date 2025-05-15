import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import os
from IPython.display import display
import logging
from tqdm import tqdm
import requests

def process_image(image):
        """
        Processes an image by converting it to RGB, resizing it,
        and normalizing pixel values.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            np.ndarray: The processed image as a NumPy array.
        """
        # Convert all images to RGB mode to ensure 3 channels
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
        return image_array

def process_local_data():
    # Get the current working directory
    current_directory = os.getcwd()
    # Construct the path to the file
    file_path = os.path.join(current_directory, 'imdb-movies-dataset.csv')  # Assuming it's in the same folder

    df = None
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        display(df.head())
        print(df.shape)
    except FileNotFoundError:
        print(f"Error: 'imdb-movies-dataset.csv' not found in {current_directory}.")
    except pd.errors.ParserError:
        print("Error: Could not read CSV file with any of the common delimiters.")

    # Identify columns related to movie posters, genres, and user ratings
    poster_column = 'Poster'
    genre_column = 'Genre'
    rating_column = 'Rating'
    
    # Handle Missing Poster URLs: Impute with placeholder
    df[poster_column].fillna("N/A", inplace=True)

    # Handle Missing Genre Values: Impute with "Unknown"
    df[genre_column].fillna("Unknown", inplace=True)

    # Clean and Standardize Genre Column
    def clean_genre(genre_str):
        genres = [g.strip() for g in genre_str.split(',')]
        return ','.join(genres)

    df[genre_column] = df[genre_column].apply(clean_genre)

    # Handle Rating Outliers: Winsorize
    lower_limit = df[rating_column].quantile(0.01)
    upper_limit = df[rating_column].quantile(0.99)
    df[rating_column] = np.clip(df[rating_column], lower_limit, upper_limit)

    # Display first few rows of the cleaned DataFrame
    display(df.head())

    # Create a random subset of 1000 rows
    df_sample = df.sample(n=500, random_state=42).copy()

    # Dropping NaN values
    df_sample = df_sample.dropna(subset=[rating_column])

    # logging for error tracking
    logging.basicConfig(level=logging.ERROR, filename='image_download_errors.log', filemode='w')

    # Create a directory to store downloaded images
    if not os.path.exists('movie_posters'):
        os.makedirs('movie_posters')

    # 1. Image Data Preparation
    image_data = []
    errors = []

    # Create a progress bar specifically designed for Jupyter notebooks
    for index, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing Images"):
        poster_url = row[poster_column]
        if poster_url != 'N/A':
            try:
                response = requests.get(poster_url, stream=True, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Check for "Not Found" errors (status code 404)
                if response.status_code == 404:
                    errors.append("Not Found")
                    continue  # Skip to the next iteration

                image = Image.open(BytesIO(response.content))
                image_array = process_image(image)
                image_data.append(image_array)

                errors.append(None)
            except requests.exceptions.RequestException as e:
                errors.append(f"RequestError: {e}")
            except Exception as e:
                errors.append(f"Error: {e}")
        else:
            # Handle N/A poster URLs
            errors.append("N/A")

    # Convert the image data list to a NumPy array
    df_sample['error'] = errors
    df_sample = df_sample[df_sample['error'].isnull()]

    image_data = np.array(image_data)

    # 2. Genre Data Preparation
    all_genres = set()
    for genres in tqdm(df_sample[genre_column].str.split(','), desc="Extracting Genres"):
        if isinstance(genres, list):  # Check if genres is a list
            all_genres.update(genres)
    all_genres = list(all_genres)
    genre_data = []
    for genres in tqdm(df_sample[genre_column].str.split(','), desc="Creating Genre Vectors"):
        if isinstance(genres, list):  # Check if genres is a list
            genre_vector = [1 if genre in genres else 0 for genre in all_genres]
        else:
            genre_vector = [0 for _ in all_genres]  # Empty vector for missing genres
        genre_data.append(genre_vector)
    genre_data = np.array(genre_data)

    # 3. Rating Data Preparation (Already numerical)
    rating_data = np.array(df_sample[rating_column])

    return image_data, genre_data, all_genres, rating_data