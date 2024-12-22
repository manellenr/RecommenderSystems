# Recommender System for Movies

## **Project Overview**

This project focuses on developing a movie recommender system using the MovieLens dataset. The system analyzes movie metadata and user ratings to suggest top-rated movies and personalized recommendations based on user preferences. It implements both content-based filtering and collaborative filtering techniques, ensuring accurate and meaningful movie suggestions.

---

## **Dataset Context**

The dataset comprises metadata for 45,000 movies listed in the Full MovieLens dataset. It includes movies released on or before July 2017 and provides a comprehensive view of movie features such as cast, crew, keywords, budget, revenue, release dates, languages, and more. Additionally, it contains 26 million ratings from 270,000 users on a scale of 1-5. You can access the dataset [here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### **Dataset Files**

- **`movies_metadata.csv`**: Core metadata for 45,000 movies, including details like budget, revenue, runtime, and vote statistics.
- **`ratings_small.csv`**: A sample of 100,000 ratings from 700 users on 9,000 movies.

---

## **System Features**

### **1. Data Loading and Cleaning**
- **Cleaning budget and revenue fields**: Handles non-numeric characters and missing values.
- **Runtime and release date processing**: Filters movies based on realistic runtime and extracts release years from dates.

### **2. Content-Based Filtering**
This approach ranks movies based on their weighted ratings, considering vote averages and vote counts. 

- **Weighted Rating Formula**:  
\[
\text{Weighted Rating} = \left( \frac{v}{v + m} \times R \right) + \left( \frac{m}{v + m} \times C \right)
\]
  Where:
  - \( v \): Number of votes for the movie  
  - \( R \): Average vote for the movie  
  - \( C \): Mean vote across all movies  
  - \( m \): Minimum votes threshold

### **3. Collaborative Filtering**
Collaborative filtering uses user-item interaction data to recommend movies based on the preferences of similar users. It includes:
- Building a **user-item rating matrix**.
- Computing **user similarity** using cosine similarity.
- Identifying **nearest neighbors** for a given user.
- Generating personalized movie recommendations from neighbors’ preferences.

---

## **Key Results**

### **Top 10 Movies by Weighted Rating**
| **Title**                      | **Weighted Rating** |
|--------------------------------|---------------------|
| Dilwale Dulhania Le Jayenge    | 8.91               |
| The Shawshank Redemption       | 8.49               |
| The Godfather                  | 8.48               |
| Your Name.                     | 8.40               |
| The Dark Knight                | 8.29               |
| Fight Club                     | 8.29               |
| Pulp Fiction                   | 8.29               |
| Schindler's List               | 8.28               |
| Whiplash                       | 8.28               |
| Spirited Away                  | 8.28               |

### **Recommended Movies for User 2**
- TMDB IDs: `[54272, 1, 33794, 69122, 520, 3593, 2571, 44555, 524, 55820]`

### **Evaluation (RMSE)**
The system achieves a Root Mean Square Error (RMSE) of **0.577**, indicating strong predictive accuracy for the recommendation model.

---

## **How to Use**

1. **Run the Script**  
   Execute `main.py` to load the data, clean it, and generate recommendations.

2. **Explore the Dataset**  
   Use the dataset to analyze movie trends or train other machine learning models.

3. **Customize Recommendations**  
   Modify the parameters for filtering movies or change the similarity calculation method for tailored results.

---

## **Future Improvements**

- **Hybrid Models**: Combine content-based and collaborative filtering for more comprehensive recommendations.
- **Incorporate NLP**: Analyze movie plots and reviews to improve recommendations.
- **Scale-Up**: Extend the system to the full 26 million ratings for better generalization.
