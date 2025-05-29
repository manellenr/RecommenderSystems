# Movie Recommendation System

## Project Overview

This project aims to develop a movie recommendation system using the MovieLens dataset. The system analyzes movie metadata as well as user ratings to suggest the highest-rated movies and provide personalized recommendations based on user preferences. It implements both content-based filtering and collaborative filtering techniques. 

The dataset contains metadata for 45,000 movies from the full MovieLens dataset. It includes movies released up to July 2017 and offers a comprehensive view of movie features such as cast, crew, keywords, budget, revenue, release dates, languages, and more. It also contains 26 million ratings from 270,000 users on a scale from 1 to 5. The data can be accessed [here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### Data

* **`movies_metadata.csv`**: Metadata for 45,000 movies, including information like budget, revenue, runtime, and voting statistics.
* **`ratings_small.csv`**: A sample of 100,000 ratings from 700 users on 9,000 movies.

## Project Steps

### 1. Data Cleaning

* Handling non-numeric characters and missing values.
* Filtering movies based on realistic runtimes and extracting release years from dates.

### 2. Content-Based Filtering

This method ranks movies based on their weighted ratings, considering both the average rating and the number of votes.

* **Weighted Rating Formula**:

$$
\text{Weighted Rating} = \left( \frac{v}{v + m} \times R \right) + \left( \frac{m}{v + m} \times C \right)
$$

where:

* \( v \): number of votes for the movie  
* \( R \): average rating of the movie  
* \( C \): mean vote across all movies  
* \( m \): minimum votes threshold

### 3. Collaborative Filtering

Collaborative filtering uses user-item interaction data to recommend movies based on similar users' preferences. It involves:

* Building a **user-item rating matrix**.
* Calculating **user similarity** using cosine similarity.
* Identifying **nearest neighbors** for a given user.
* Generating personalized recommendations from neighbors’ preferences.

---

## Results

### Top 10 Movies by Weighted Rating

| **Title**                   | **Weighted Rating** |
| --------------------------- | ------------------- |
| Dilwale Dulhania Le Jayenge | 8.91                |
| The Shawshank Redemption    | 8.49                |
| The Godfather               | 8.48                |
| Your Name.                  | 8.40                |
| The Dark Knight             | 8.29                |
| Fight Club                  | 8.29                |
| Pulp Fiction                | 8.29                |
| Schindler's List            | 8.28                |
| Whiplash                    | 8.28                |
| Spirited Away               | 8.28                |

## Conclusion

To further enhance the project, I can combine content-based and collaborative filtering for more comprehensive recommendations. Additionally, integrating NLP to analyze movie plots and reviews could refine recommendations. Alternatively, more recent recommendation models like Two Towers Embeddings could be used for improved performance.
