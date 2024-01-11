# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model
car_price_model = joblib.load("C:\\Users\\ADMIN\\Desktop\\python\\ROHIT\\car_price\\build-model\\lr_model.joblib")


# Encoding dictionaries
fuel_type_stoi = {
    "Petrol": 1,
    "Diesel": 2,
    "CNG": 3,
    "LPG": 4,
    "Electric": 5,
    "CNG + CNG": 6,
    "Hybrid": 7,
    "Petrol + CNG": 8,
    "Petrol + LPG": 9,
}
owner_stoi = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "UnRegistered Car": 4,
    "4 or More": 5,
}

make_stoi = {
    "Honda": 7,
    "Maruti Suzuki": 18,
    "Hyundai": 8,
    "Toyota": 29,
    "BMW": 1,
    "Skoda": 26,
    "Nissan": 22,
    "Renault": 24,
    "Tata": 28,
    "Volkswagen": 30,
    "Ford": 6,
    "Mercedes-Benz": 20,
    "Audi": 0,
    "Mahindra": 17,
    "MG": 15,
    "Jeep": 11,
    "Porsche": 23,
    "Kia": 12,
    "Land Rover": 13,
    "Volvo": 31,
    "Maserati": 19,
    "Jaguar": 10,
    "Isuzu": 9,
    "Ferrari": 4,
    "Mitsubishi": 21,
    "Datsun": 3,
    "MINI": 16,
    "Chevrolet": 2,
    "Ssangyong": 27,
    "Fiat": 5,
    "Rolls-Royce": 25,
    "Lexus": 14,
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        # Get input data from the request form
        make = request.form["Make"]
        owner = request.form["Owner"]
        year = float(int(request.form["Year"]))
        kilometer = float(request.form["Kilometer"])
        fuel_type = request.form["FuelType"]
        # engine_cc = int(request.form['Engine_cc'])

        # Handle missing or unexpected input
        make_encoded = make_stoi.get(make)
        owner_encoded = owner_stoi.get(owner)
        fuel_type_encoded = fuel_type_stoi.get(fuel_type)

        if None in [make_encoded, owner_encoded, fuel_type_encoded]:
            return render_template(
                "index.html", error_msg="Invalid input. Please check your input values."
            )

        # Create a dictionary with input data
        model_input = [make_encoded, owner_encoded, year, kilometer, fuel_type_encoded]

        # Make a prediction using the model
        prediction = car_price_model.predict([model_input])
        prediction_text = '%.3f' % prediction[0]

        # Return the prediction as a response
        return render_template("index.html", prediction_text=prediction_text, error_msg=None)


if __name__ == "__main__":
    app.run(debug=True)
