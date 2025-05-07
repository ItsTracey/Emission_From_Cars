# Emission_From_Cars

# CO₂ Emissions Predictor (CLI)

This command-line Python app predicts a vehicle's CO₂ emissions based on three features: engine size, number of cylinders, and fuel consumption (L/100 km). It uses a linear regression model trained on real automotive data from Canada.

---

## Features

- Train/test split to evaluate model performance  
- Predicts CO₂ emissions based on user input  
- Visualizes how each input contributes to the final prediction  
- Input validation to improve user experience  
- Clean and beginner-friendly codebase

---

## How It Works

1. Loads and processes a dataset of vehicles and their emissions
2. Trains a `LinearRegression` model using three input features:
   - Engine Size (L)
   - Cylinders
   - Fuel Consumption (L/100 km)
3. Takes user input via command line
4. Outputs the predicted CO₂ emissions
5. Optionally shows a bar chart explaining each feature’s contribution to the prediction

---

## Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib

---

## Getting Started

### 1. Clone the repository
git clone https://github.com/yourusername/CO2-Emissions-Predictor.git
cd CO2-Emissions-Predictor

### 2. Install dependencies
Make sure you have Python 3 installed, then:
pip install -r requirements.txt
Or manually install:
pip install pandas numpy matplotlib scikit-learn

### 3. Run the app
python Emmisions_From_Cars.py

---

## Dataset
This app uses the "CO2 Emissions_Canada.csv" dataset which includes car specs and their corresponding CO₂ emissions.

---

## Example Input

What is the size of your vehicle engine (in Litres)? 2.0
How many cylinders does your vehicle have? 4
What is the fuel consumption (L/100 km)? 8.7

---
