from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load the dataset and clean it
data = pd.read_csv('house_prices_india.csv')
data['Price_per_sq_ft'] = data['Price_per_sq_ft'].str.replace(',', '').astype(float)

# Prepare the model
X = data[['Appartment_type', 'Price_per_sq_ft', 'Area', 'Status', 'Location']]
y = data['Price']

# Define categorical and numerical features
categorical_features = ['Appartment_type', 'Status', 'Location']
numerical_features = ['Price_per_sq_ft', 'Area']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# Initialize Flask App
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    locations = data['Location'].unique()
    apartment_types = data['Appartment_type'].unique()
    return render_template('index.html', locations=locations, apartment_types=apartment_types)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form['Location']
    apartment_type = request.form['Appartment_type']
    area = float(request.form['Area'])
    price_per_sq_ft = float(request.form['Price_per_sq_ft'])

    # Prepare input data for prediction
    input_data = pd.DataFrame([[apartment_type, price_per_sq_ft, area, 'Ready to move', location]], columns=['Appartment_type', 'Price_per_sq_ft', 'Area', 'Status', 'Location'])

    # Make the prediction
    predicted_price = model.predict(input_data)[0]

    return render_template('index.html', predicted_price=predicted_price, locations=data['Location'].unique(), apartment_types=data['Appartment_type'].unique())

if __name__ == '__main__':
    app.run(debug=True, port=5001)


