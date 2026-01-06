import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('parkinsons.csv')

X = data[['PPE', 'spread1', 'MDVP:Fo(Hz)']]
y = data['status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_val_scaled, y_val)
print(accuracy)

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
