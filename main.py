import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import yaml

# 1. Load the dataset
df = pd.read_csv("parkinsons.csv")

# 2. Select features and target
X = df[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]]
y = df["status"]
selected_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]

print("X sample:")
print(X.head())
print("\nY

