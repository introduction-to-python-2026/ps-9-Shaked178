import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('parkinsons.csv')

X = data.drop(['status', 'name'], axis=1, errors='ignore')
y = data['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Done")


model_path = "model.joblib"
target = "status"
features = ['PPE', 'spread1']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)


model.fit(X_train, y_train)
accuracy = model.score(X_val , y_val)
print(f"Accuracy: {accuracy}")
