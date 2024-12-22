import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Function to load and clean the data
def load_and_clean_data(file_path):
    colnames = ["title", "release_date", "budget", "revenue", "runtime", "genres", "vote_count", "vote_average"]
    
    df = pd.read_csv(file_path, usecols=colnames)
    
    # Cleaning the 'budget' and 'revenue' columns to ensure they are numeric
    # Remove any non-numeric characters (such as file paths or image URLs)
    df["budget"] = df["budget"].replace({'\$': '', ',': '', '/.*': ''}, regex=True)

    # Convert the budget column to numeric, replacing any problematic values with NaN
    df["budget"] = pd.to_numeric(df["budget"], errors='coerce')

    # Handling missing values
    df['budget'].fillna(df['budget'].median(), inplace=True)
    df['revenue'].fillna(df['revenue'].median(), inplace=True)
    df['runtime'].fillna(df['runtime'].median(), inplace=True)
    df.dropna(subset=['title', 'release_date'], inplace=True)
    
    # Extracting the release year
    df["release_date"] = pd.to_datetime(df["release_date"], format='%Y-%m-%d', errors="coerce")
    df["year"] = pd.DatetimeIndex(df["release_date"]).year
    
    return df

# Load and clean the data
df = load_and_clean_data("movies_metadata.csv")
print(df.columns)
df.head(3)

# Function to filter movies based on runtime and vote count conditions
def filter_movies(data, min_runtime=45, max_runtime=300, quantile=0.80):
    if "vote_count" not in data.columns:
        raise KeyError("'vote_count' column is missing in the dataset.")
    
    m = data["vote_count"].quantile(quantile)  
    filtered_data = data[(data["runtime"] >= min_runtime) & (data["runtime"] <= max_runtime)]
    filtered_data = filtered_data[filtered_data["vote_count"] >= m]
    return filtered_data

# Function to calculate the weighted rating for movies
def calculate_weighted_rating(data, min_votes_threshold=50):
    # Average vote across all movies
    C = data['vote_average'].sum() / len(data)
    
    # Calculating the weighted rating
    weighted_ratings = []
    for _, row in data.iterrows():
        v = row['vote_count']
        R = row['vote_average']
        weighted_rating = ((v / (v + min_votes_threshold)) * R) + ((min_votes_threshold / (v + min_votes_threshold)) * C)
        weighted_ratings.append(weighted_rating)
    
    data['Weighted Rating'] = weighted_ratings
    return data

# Function to get the top N movies based on weighted rating
def get_top_n_movies(data, n=10):
    return data.sort_values(by="Weighted Rating", ascending=False).head(n)[['title', 'Weighted Rating']]

# Filter the movies and calculate the weighted ratings
filtered_data = filter_movies(df)
rated_movies = calculate_weighted_rating(filtered_data)
top_movies = get_top_n_movies(rated_movies)
print(top_movies)

# Function to create the user-item rating matrix
def create_user_item_matrix(ratings_df):
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item_matrix

# Function to compute user similarity matrix
def compute_user_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)

# Function to get the nearest neighbors of a given user
def get_neighbors(user_similarity_matrix, user_index, n=5):
    similar_users = user_similarity_matrix[user_index]
    similar_users_sorted = np.argsort(similar_users)[::-1] 
    return similar_users_sorted[:n]

# Function to recommend movies based on nearest neighbors
def recommend_movies(user_index, user_item_matrix, user_similarity_matrix, n_neighbors=5, top_k=10):
    neighbors_indices = get_neighbors(user_similarity_matrix, user_index, n_neighbors)
    recommended_movies = []
    for neighbor_index in neighbors_indices:
        # Get the movies rated by the neighbor
        neighbor_ratings = user_item_matrix.iloc[neighbor_index]
        # Filter the movies not yet rated by the current user
        movies_to_recommend = neighbor_ratings[neighbor_ratings > 0].index
        recommended_movies.extend(movies_to_recommend)
    
    # Return the most popular recommended movies (unique)
    recommended_movies = list(set(recommended_movies))  # Remove duplicates
    return recommended_movies[:top_k]

# Load ratings data
ratings = pd.read_csv("ratings_small.csv")

# Create user-item matrix and compute similarity
user_item_matrix = create_user_item_matrix(ratings)
user_similarity_matrix = compute_user_similarity(user_item_matrix)

# Recommend movies for user 2
recommended_movies = recommend_movies(user_index=2, user_item_matrix=user_item_matrix, 
                                      user_similarity_matrix=user_similarity_matrix)
print("Recommended Movies:", recommended_movies)

# Function to evaluate the performance using RMSE
def evaluate_rmse(predictions, actual_ratings):
    return np.sqrt(mean_squared_error(actual_ratings, predictions))

# Example of evaluating the performance of the recommender system
# Here, we assume 'predictions' is an array of predicted ratings and 'actual_ratings' is the true ratings
rmse = evaluate_rmse(predictions=np.array([4, 5, 3]), actual_ratings=np.array([4, 4, 3]))
print(f"RMSE: {rmse}")
