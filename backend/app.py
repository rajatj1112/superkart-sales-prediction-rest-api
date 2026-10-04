# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model pipeline
model = joblib.load("superkart_model.joblib")


# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    Handles GET requests to the root URL ('/') and returns a welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"


# Define an endpoint for single prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    """
    Handles POST requests to '/v1/predict'. Expects a JSON payload with the
    product and store details and returns the predicted sales as JSON.
    """
    # Get the JSON data from the request body
    product_data = request.get_json()

    # Extract the relevant features from the JSON data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make the prediction using the loaded pipeline
    prediction = model.predict(input_data)[0]

    # Convert the prediction to a Python float (NumPy types are not JSON serializable)
    prediction = round(float(prediction), 2)

    # Return the predicted sales as a JSON response
    return jsonify({'Predicted Sales (in dollars)': prediction})


# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    Handles POST requests to '/v1/predictbatch'. Expects a CSV file containing
    multiple product/store records and returns the predicted sales for each row.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all rows in the DataFrame
    predictions = model.predict(input_data).tolist()

    # Round the predictions and build an index -> prediction dictionary
    predictions = [round(float(pred), 2) for pred in predictions]
    output_dict = dict(zip(input_data.index.astype(str), predictions))

    # Return the predictions dictionary as a JSON response
    return output_dict


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(debug=True)
