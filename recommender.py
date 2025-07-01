import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load CSV File
try:
    df = pd.read_csv("C:/Users/sanja/OneDrive/Desktop/python_practice/movie_recomender/movies.csv")
except FileNotFoundError:
    print("⚠️ 'movies.csv' file not found in the current directory.")
    exit()

# Step 2: Combine 'genres' and 'overview' into one feature
df['combined'] = df['genres'].fillna('') + " " + df['overview'].fillna('')

# Step 3: Convert combined text into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

# Step 4: Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Recommend similar movies
def recommend(movie_title, num=5):
    if movie_title not in df['title'].values:
        print("❌ Movie not found in dataset.")
        return

    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]

    print(f"\n🎬 Top {num} movies similar to '{movie_title}':")
    for i, (index, score) in enumerate(sim_scores, start=1):
        print(f"{i}. {df.iloc[index]['title']}")

# Step 6: Run the recommender in CLI
if __name__ == "__main__":
    print("🎥 Movie Recommendation System")
    movie_input = input("Enter a movie title: ").strip()
    recommend(movie_input)
