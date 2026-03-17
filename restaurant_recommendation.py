import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('restaurant_dataset.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Preprocessing
# Handle missing values
print("\nMissing values:")
print(df.isnull().sum())

# For simplicity, drop rows with missing cuisines or price range
df = df.dropna(subset=['Cuisines', 'Price range'])

# Fill missing ratings with mean
df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].mean())

# Encode categorical variables
# Cuisines: split by comma and encode
df['Cuisines'] = df['Cuisines'].str.split(', ')

mlb = MultiLabelBinarizer()
cuisine_encoded = mlb.fit_transform(df['Cuisines'])
cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb.classes_, index=df.index)

# Price range is already numerical 1-4
# Aggregate rating is numerical
# Let's also include city, but for simplicity, maybe not

# Combine features
features = pd.concat([cuisine_df, df[['Price range', 'Aggregate rating']]], axis=1)

# Normalize numerical features
scaler = StandardScaler()
features[['Price range', 'Aggregate rating']] = scaler.fit_transform(features[['Price range', 'Aggregate rating']])

print("\nFeatures shape:", features.shape)

# Content-based filtering
def recommend_restaurants(user_preferences, top_n=5):
    """
    user_preferences: dict with 'cuisines': list of preferred cuisines, 'price_range': int 1-4
    """
    # Create user vector
    user_vector = np.zeros(features.shape[1])
    
    # Set cuisine preferences
    for cuisine in user_preferences.get('cuisines', []):
        if cuisine in mlb.classes_:
            idx = list(mlb.classes_).index(cuisine)
            user_vector[idx] = 1
    
    # Set price range (normalized)
    price_idx = features.columns.get_loc('Price range')
    user_vector[price_idx] = scaler.transform([[user_preferences.get('price_range', 2), 0]])[0][0]
    
    # Rating: assume user likes high rated, but for simplicity, set to mean or something
    rating_idx = features.columns.get_loc('Aggregate rating')
    user_vector[rating_idx] = 0  # or some value
    
    # Compute similarity
    similarities = cosine_similarity([user_vector], features)[0]
    
    # Get top similar
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating', 'City']]
    recommendations['Similarity'] = similarities[top_indices]
    
    return recommendations

# Test the system
print("\nTesting with sample preferences:")
sample_prefs = {'cuisines': ['Japanese', 'Chinese'], 'price_range': 3}
recs = recommend_restaurants(sample_prefs, top_n=5)
print(recs)

# Another test
sample_prefs2 = {'cuisines': ['Italian', 'Pizza'], 'price_range': 2}
recs2 = recommend_restaurants(sample_prefs2, top_n=5)
print("\nAnother test:")
print(recs2)

# Evaluation: For simplicity, since no ground truth, just check if recommendations match preferences
def evaluate_recommendations(user_prefs, recommendations):
    matches = 0
    for idx, row in recommendations.iterrows():
        cuisines = set(row['Cuisines'])
        if any(c in cuisines for c in user_prefs['cuisines']):
            matches += 1
        if row['Price range'] == user_prefs['price_range']:
            matches += 1
    return matches / (len(recommendations) * 2)  # proportion of matches

eval1 = evaluate_recommendations(sample_prefs, recs)
print(f"\nEvaluation score for first test: {eval1:.2f}")

eval2 = evaluate_recommendations(sample_prefs2, recs2)
print(f"Evaluation score for second test: {eval2:.2f}")