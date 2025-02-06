import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify

# Load the CSV file
data = pd.read_csv('D:/WebiSoftTech/DECISION TREE/Salaries/salaries.csv')

# Encode categorical variables
label_encoders = {}
for column in ['company', 'job', 'degree']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['company', 'job', 'degree']]
y = data['salary_more_than_100k']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Setup Flask Application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Check for required keys in the request data
    required_keys = ['company', 'job', 'degree']
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'Missing key: {key}'}), 400

    # Encode the input data
    try:
        company_encoded = label_encoders['company'].transform([data['company']])[0]
        job_encoded = label_encoders['job'].transform([data['job']])[0]
        degree_encoded = label_encoders['degree'].transform([data['degree']])[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make prediction
    prediction = model.predict([[company_encoded, job_encoded, degree_encoded]])

    return jsonify({'salary_more_than_100k': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)