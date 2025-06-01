import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def load_and_clean_data(file_path):
    colnames = [
        "title", "release_date", "budget", "revenue", "runtime", "genres", 
        "vote_count", "vote_average", "popularity"
    ]
    
    df = pd.read_csv(file_path, usecols=colnames, low_memory=False)

    df["budget"] = df["budget"].replace({'\$': '', ',': '', '/.*': ''}, regex=True)
    df["budget"] = pd.to_numeric(df["budget"], errors='coerce')

    df["revenue"] = pd.to_numeric(df["revenue"], errors='coerce')

    df["runtime"] = pd.to_numeric(df["runtime"], errors='coerce')

    df["vote_count"] = pd.to_numeric(df["vote_count"], errors='coerce')

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors='coerce')

    df["release_date"] = pd.to_datetime(df["release_date"], format='%Y-%m-%d', errors="coerce")
    df["year"] = pd.DatetimeIndex(df["release_date"]).year

    df['budget'].fillna(df['budget'].median(), inplace=True)
    df['revenue'].fillna(df['revenue'].median(), inplace=True)
    df['runtime'].fillna(df['runtime'].median(), inplace=True)
    df['vote_count'].fillna(0, inplace=True)
    df['vote_average'].fillna(df['vote_average'].median(), inplace=True)

    df.dropna(subset=['title', 'release_date'], inplace=True)
    
    return df

df = load_and_clean_data("data/movies_metadata.csv")

def filter_movies(data, min_runtime=45, max_runtime=300, quantile=0.80):
    m = data["vote_count"].quantile(quantile)
    filtered_data = data[(data["runtime"] >= min_runtime) & 
                         (data["runtime"] <= max_runtime) & 
                         (data["vote_count"] >= m)]
    return filtered_data, m

def calculate_weighted_rating(data, m):
    C = data['vote_average'].mean()
    v = data['vote_count']
    R = data['vote_average']
    data['Weighted Rating'] = ((v / (v + m)) * R) + ((m / (v + m)) * C)
    return data

def get_top_n_movies(data, n=10):
    return data.sort_values(by="Weighted Rating", ascending=False).head(n)[['title', 'Weighted Rating']]

def create_user_item_matrix(ratings_df):
    return ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

def compute_user_similarity(user_item_matrix):
    return cosine_similarity(user_item_matrix)

def get_neighbors(user_similarity_matrix, user_index, n=5):
    similar_users = user_similarity_matrix[user_index]
    similar_users_sorted = np.argsort(similar_users)[::-1]  
    return similar_users_sorted[1:n+1]  

def recommend_movies(user_index, user_item_matrix, user_similarity_matrix, n_neighbors=5, top_k=10):
    neighbors_indices = get_neighbors(user_similarity_matrix, user_index, n_neighbors)
    recommended_movies = []
    for neighbor_index in neighbors_indices:
        neighbor_ratings = user_item_matrix.iloc[neighbor_index]
        movies_to_recommend = neighbor_ratings[neighbor_ratings > 0].index
        recommended_movies.extend(movies_to_recommend)
    
    recommended_movies = list(set(recommended_movies))
    return recommended_movies[:top_k]

def evaluate_rmse(predictions, actual_ratings):
    return np.sqrt(mean_squared_error(actual_ratings, predictions))

filtered_data, m = filter_movies(df)
rated_movies = calculate_weighted_rating(filtered_data, m)
top_movies = get_top_n_movies(rated_movies)
print("Top 10 Movies:")
print(top_movies)

ratings = pd.read_csv("data/ratings_small.csv")
user_item_matrix = create_user_item_matrix(ratings)
user_similarity_matrix = compute_user_similarity(user_item_matrix)

recommended_movies = recommend_movies(user_index=2, user_item_matrix=user_item_matrix, 
                                      user_similarity_matrix=user_similarity_matrix)
print("Recommended Movies for User 2:", recommended_movies)

rmse = evaluate_rmse(predictions=np.array([4, 5, 3]), actual_ratings=np.array([4, 4, 3]))
print(f"RMSE: {rmse:.2f}")
