from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('D:/WebiSoftTech/DECISION TREE/Cricket1/cricket1.csv')
X = data.drop('Play Cricket', axis=1)  # Features
y = data['Play Cricket']  # Target variable

# Convert categorical variables to numerical
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json(force=True)
    features = [data['Outlook'], data['Temperature'], data['Humidity'], data['Windy']]

    # Convert input features to a DataFrame and apply one-hot encoding
    input_df = pd.DataFrame([features], columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure the input has the same number of columns as the training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Standardize the input features
    features_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = knn.predict(features_scaled)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)