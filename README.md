# moive
import joblib

# Save the trained model
joblib.dump(model, 'movie_recommendation_model.joblib')

# Load the model later
# loaded_model = joblib.load('movie_recommendation_model.joblib')

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Assuming you have a ratings DataFrame with columns 'userId', 'movieId', 'rating'
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Initialize the SVD algorithm
model = SVD(n_factors=50, random_state=42)

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.predict(uid=testset[0][0], iid=testset[0][1]) # Predict rating for a specific user-item pair
test_predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(test_predictions)
accuracy.mae(test_predictions)

# To get top N recommendations for a user:
def get_top_n_recommendations(predictions, n=10):
    # ... (implementation to group predictions by user and sort by estimated rating)
    pass

# Example of getting recommendations for a specific user
top_n = get_top_n_recommendations(predictions, n=10)
print(top_n)
