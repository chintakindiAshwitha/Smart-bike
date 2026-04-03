
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------------------------
# BIKE USAGE MODEL
# ---------------------------

bike_data = {
    "hour": [6, 8, 9, 12, 17, 18, 20, 22],
    "temp": [20, 25, 28, 30, 27, 26, 24, 22],
    "weather": [0, 1, 1, 0, 2, 2, 1, 0],
    "traffic": [1, 2, 2, 1, 2, 2, 1, 0],
    "rent": [1, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(bike_data)

X = df[["hour", "temp", "weather", "traffic"]]
y = df["rent"]

bike_model = RandomForestClassifier()
bike_model.fit(X, y)

pickle.dump(bike_model, open("bike_model.pkl", "wb"))

# ---------------------------
# TRAFFIC MODEL
# ---------------------------

traffic_data = {
    "hour": [6, 8, 9, 12, 17, 18, 20, 22],
    "weather": [0, 1, 1, 0, 2, 2, 1, 0],
    "traffic": [1, 2, 2, 1, 2, 2, 1, 0]
}

df2 = pd.DataFrame(traffic_data)

X2 = df2[["hour", "weather"]]
y2 = df2["traffic"]

traffic_model = RandomForestClassifier()
traffic_model.fit(X2, y2)

pickle.dump(traffic_model, open("traffic.pkl", "wb"))

print("✅ Both models trained and saved!")