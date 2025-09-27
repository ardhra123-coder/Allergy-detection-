import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_allergy_model.pkl")

# Feature order should match training data
feature_names = [
    "Age", "Gender", "Family_History", "Previous_Reaction", 
    "Symptoms", "Food_Type", "Food_Frequency", 
    "Medical_Conditions", "IgE_Levels", "Severity_Score"
]

# For categorical encoding, we need to match training encodings
# You should save encoders during training; for demo, let's define mapping manually
gender_map = {"Male": 1, "Female": 0, "Other": 2}
family_history_map = {"No": 0, "Yes": 1}
previous_reaction_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
symptoms_map = {"No symptoms": 0, "Nausea": 1, "Skin rash": 2, "Swelling": 3}
food_type_map = {"Eggs": 0, "Gluten": 1, "Nuts": 2, "Milk": 3, "Seafood": 4}
medical_conditions_map = {"None": 0, "Asthma": 1, "Diabetes": 2, "Hypertension": 3}

def get_input():
    print("\nEnter patient details for allergy prediction:\n")
    
    age = int(input("Age: "))
    gender = gender_map[input("Gender (Male/Female/Other): ")]
    family_history = family_history_map[input("Family History (Yes/No): ")]
    prev_reaction = previous_reaction_map[input("Previous Reaction (None/Mild/Moderate/Severe): ")]
    symptoms = symptoms_map[input("Symptoms (No symptoms/Nausea/Skin rash/Swelling): ")]
    food_type = food_type_map[input("Food Type (Eggs/Gluten/Nuts/Milk/Seafood): ")]
    food_freq = int(input("Food Frequency (times per month): "))
    med_cond = medical_conditions_map[input("Medical Conditions (None/Asthma/Diabetes/Hypertension): ")]
    ige = float(input("IgE Levels: "))
    severity = int(input("Severity Score (1-10): "))

    return [age, gender, family_history, prev_reaction, symptoms, 
            food_type, food_freq, med_cond, ige, severity]

def main():
    user_input = get_input()
    user_input = np.array(user_input).reshape(1, -1)

    prediction = model.predict(user_input)[0]

    if prediction == 1:
        print("\n🔴 The patient is likely ALLERGIC.")
    else:
        print("\n🟢 The patient is NOT allergic.")

if __name__ == "__main__":
    main()
