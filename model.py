import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------------------------
# BIKE USAGE MODEL (IMPROVED)
# ---------------------------

bike_data = {
    "hour":    [6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22],
    "temp":    [18, 20, 25, 28, 30, 32, 35, 33, 29, 27, 26, 24, 23, 22],
    "weather": [0, 0, 1, 1, 2, 0, 2, 1, 2, 2, 1, 1, 0, 0],  # 0=Sunny,1=Cloudy,2=Rainy
    "traffic": [0, 1, 2, 2, 1, 1, 0, 1, 2, 2, 1, 1, 0, 0],  # 0=Low,1=Medium,2=High
    "rent":    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1]   # 1=Good,0=Bad
}

df = pd.DataFrame(bike_data)

X = df[["hour", "temp", "weather", "traffic"]]
y = df["rent"]

bike_model = RandomForestClassifier(n_estimators=100, random_state=42)
bike_model.fit(X, y)

pickle.dump(bike_model, open("bike_model.pkl", "wb"))

# ---------------------------
# TRAFFIC MODEL (IMPROVED)
# ---------------------------

traffic_data = {
    "hour":    [6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22],
    "weather": [0, 0, 1, 1, 2, 0, 2, 1, 2, 2, 1, 1, 0, 0],
    "traffic": [0, 1, 2, 2, 1, 1, 0, 1, 2, 2, 1, 1, 0, 0]
}

df2 = pd.DataFrame(traffic_data)

X2 = df2[["hour", "weather"]]
y2 = df2["traffic"]

traffic_model = RandomForestClassifier(n_estimators=100, random_state=42)
traffic_model.fit(X2, y2)

pickle.dump(traffic_model, open("traffic.pkl", "wb"))

print("✅ Both models trained and saved!")