import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the trained model
with open("car_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Flask server is running! Use /predict for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get input from frontend
        print("Received Data:", data)  # Debugging

        # Define expected features
        feature_names = [            
            "year_of_manufacture", "kms_driven", "fuel_type_Diesel",
            "fuel_type_Electric", "fuel_type_Hybrid", "fuel_type_LPG",
            "fuel_type_Petrol", "city_Ambattur", "city_Bangalore",
            "city_Chennai", "city_Delhi", "city_Faridabad", "city_Gurgaon",
            "city_Hyderabad", "city_Kolkata", "city_Mumbai", "city_Noida",
            "city_Pallikarnai", "city_Poonamallee", "city_Pune", "city_Thane",
            "city_Thiruvallur", "car_age"
        ]

        # Extract values safely with .get() (default to 0 if missing)
        input_features = [data.get(feature, 0) for feature in feature_names]

        # Ensure all expected features are present
        if len(input_features) != len(feature_names):
            return jsonify({"error": "Missing features in request"}), 400

        # Make prediction
        prediction = model.predict([input_features])
        return jsonify({"predicted_price": round(prediction[0], 2)})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
