import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('CO2 Emissions_Canada.csv')
Engine_arr = np.array(df['Engine Size(L)'])
Cylinder_arr = np.array(df['Cylinders'])
Fuel_Consumption_arr = np.array(df['Fuel Consumption Comb (L/100 km)'])
CO2_Emissions_arr = np.array(df['CO2 Emissions(g/km)'])

# plt.scatter(Engine_arr, CO2_Emissions_arr)
# plt.title('Engine Size  vs CO₂ Emissions')
# plt.xlabel('Engine Size')
# plt.ylabel('CO₂ Emissions (g/km)')
# plt.show()

# plt.scatter(Cylinder_arr, CO2_Emissions_arr)
# plt.title('Cylinders vs CO₂ Emissions')
# plt.xlabel('Cylinders')
# plt.ylabel('CO₂ Emissions (g/km)')
# plt.show()

# plt.scatter(Fuel_Consumption_arr, CO2_Emissions_arr)
# plt.title('Fuel Consumption vs CO₂ Emissions')
# plt.xlabel('Fuel Consumption')
# plt.ylabel('CO₂ Emissions (g/km)')
# plt.show()


x = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
y = df['CO2 Emissions(g/km)']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model = model.fit(x_train, y_train)
score = model.score(x_test, y_test)
predict = model.predict(x_test)



class GetUserInput:
    def __init__(self):
        self.engine = None
        self.cylinder = None
        self.fuel_consumption = None

    def collect_input(self):
        self.engine = float(input("What is the size of your vehicle engine (in Litres)? ").replace(",", "."))
        self.cylinder = int(input("How many cylinders does your vehicle have? "))
        self.fuel_consumption = float(input("What is the fuel consumption (L/100 km)? ").replace(",", "."))

    def get_input_data(self):
        data = [[self.engine, self.cylinder, self.fuel_consumption]]
        columns = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
        return pd.DataFrame(data, columns=columns)


class EmissionPredictor:
    def __init__(self, model):
        self.model = model

    def predict_emission(self, user_input):
        prediction = self.model.predict(user_input)
        return prediction[0]

    def explain_prediction(self, user_input):
        coefs = self.model.coef_
        contributions = [coefs[i] * user_input[0][i] for i in range(len(coefs))]
        features = ['Engine Size', 'Cylinders', 'Fuel Consumption']

        plt.bar(features, contributions)
        plt.title("Feature Contributions to Predicted CO₂ Emissions")
        plt.ylabel("Contribution (g/km)")
        plt.show()

def main():
    
    print("🚗 Welcome to the CO₂ Emissions Predictor!")
    predictor = EmissionPredictor(model)

    while True:
        user = GetUserInput()
        user.collect_input()
        input_data = user.get_input_data()

        result = predictor.predict_emission(input_data)
        print(f"\n🌿 Estimated CO₂ Emissions: {result:.2f} g/km")

        view_explanation = input("📊 Would you like to see how each feature contributed? (yes/no): ").strip().lower()
        if view_explanation in ['yes', 'y']:
            predictor.explain_prediction(input_data)

        again = input("\n🔁 Would you like to predict another car? (yes/no): ").strip().lower()
        if again not in ['yes', 'y']:
            print("👋 Thanks for using the CO₂ Emissions Predictor. Goodbye!")
            break

if __name__ == "__main__":
    main()

