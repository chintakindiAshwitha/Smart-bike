
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("hour.csv")

# Select required columns
data = data[['hr', 'temp', 'weathersit', 'cnt']]

# Convert target into classification (Good / Avoid)
data['decision'] = data['cnt'].apply(lambda x: 1 if x > 200 else 0)

# Features and target
X = data[['hr', 'temp', 'weathersit']]
y = data['decision']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")