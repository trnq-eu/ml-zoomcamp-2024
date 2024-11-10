import numpy as np
import pandas as pd
import requests
import os

def generate_random_patient():
    """
    Generate a patient with random but plausible values using stats from the original dataset
    """

    # Helper function to generate random data
    def random_value(mean, std, min_val, max_val):
        while True:
            value = np.random.normal(mean, std)
            if min_val <= value <= max_val:
                return value

    patient = {
        'Age': int(random_value(54.44, 20.55, 20, 90)),
        'Gender': np.random.choice([0, 1], p=[0.48, 0.52]),
        'Ethnicity': np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.1, 0.1]),
        'SocioeconomicStatus': np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3]),
        'EducationLevel': np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]),
        'BMI': random_value(27.62, 7.29, 15.03, 40),
        'Smoking': np.random.choice([0, 1], p=[0.71, 0.29]),
        'AlcoholConsumption': random_value(9.97, 5.80, 0.02, 20),
        'PhysicalActivity': random_value(5.02, 2.87, 0, 10),
        'DietQuality': random_value(5.03, 2.87, 0, 10),
        'SleepQuality': random_value(6.94, 1.70, 4, 10),
        'FamilyHistoryKidneyDisease': np.random.choice([0, 1], p=[0.86, 0.14]),
        'FamilyHistoryHypertension': np.random.choice([0, 1], p=[0.70, 0.30]),
        'FamilyHistoryDiabetes': np.random.choice([0, 1], p=[0.74, 0.26]),
        'PreviousAcuteKidneyInjury': np.random.choice([0, 1], p=[0.89, 0.11]),
        'UrinaryTractInfections': np.random.choice([0, 1], p=[0.79, 0.21]),
        'SystolicBP': random_value(134.39, 25.77, 90, 179),
        'DiastolicBP': random_value(89.31, 17.35, 60, 119),
        'FastingBloodSugar': random_value(132.53, 36.56, 70.04, 200),
        'HbA1c': random_value(6.98, 1.73, 4, 10),
        'SerumCreatinine': random_value(2.75, 1.32, 0.50, 5),
        'BUNLevels': random_value(27.58, 12.81, 5, 50),
        'GFR': random_value(66.83, 30.05, 15.11, 120),
        'ProteinInUrine': random_value(2.49, 1.45, 0, 5),
        'ACR': random_value(149.88, 86.85, 0.18, 300),
        'SerumElectrolytesSodium': random_value(139.97, 2.91, 135, 145),
        'SerumElectrolytesPotassium': random_value(4.51, 0.58, 3.5, 5.5),
        'SerumElectrolytesCalcium': random_value(9.49, 0.57, 8.5, 10.5),
        'SerumElectrolytesPhosphorus': random_value(3.51, 0.58, 2.5, 4.5),
        'HemoglobinLevels': random_value(13.93, 2.34, 10, 18),
        'CholesterolTotal': random_value(224.25, 43.67, 150, 300),
        'CholesterolLDL': random_value(125.04, 42.65, 50, 200),
        'CholesterolHDL': random_value(60.75, 23.17, 20, 100),
        'CholesterolTriglycerides': random_value(224.80, 100.32, 50, 400),
        'ACEInhibitors': np.random.choice([0, 1], p=[0.70, 0.30]),
        'Diuretics': np.random.choice([0, 1], p=[0.68, 0.32]),
        'NSAIDsUse': random_value(5.01, 2.87, 0, 10),
        'Statins': np.random.choice([0, 1], p=[0.62, 0.38]),
        'AntidiabeticMedications': np.random.choice([0, 1], p=[0.80, 0.20]),
        'Edema': np.random.choice([0, 1], p=[0.80, 0.20]),
        'FatigueLevels': random_value(5.02, 2.90, 0, 10),
        'NauseaVomiting': random_value(3.48, 1.99, 0, 7),
        'MuscleCramps': random_value(3.53, 2.03, 0, 7),
        'Itching': random_value(5.05, 2.88, 0, 10),
        'QualityOfLifeScore': random_value(49.73, 27.83, 0, 100),
        'HeavyMetalsExposure': np.random.choice([0, 1], p=[0.96, 0.04]),
        'OccupationalExposureChemicals': np.random.choice([0, 1], p=[0.90, 0.10]),
        'WaterQuality': np.random.choice([0, 1], p=[0.80, 0.20]),
        'MedicalCheckupsFrequency': random_value(2.00, 1.14, 0, 4),
        'MedicationAdherence': random_value(4.95, 2.87, 0, 10),
        'HealthLiteracy': random_value(5.14, 2.90, 0, 10)
    }

    # Convert numpy types to native Python types
    for key, value in patient.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            patient[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            patient[key] = float(value)
        elif isinstance(value, np.bool_):
            patient[key] = bool(value)

    # Round values to 2 decimals
    for key, value in patient.items():
        if isinstance(value, float):
            patient[key] = round(value, 2)

    return patient

# Function to generate multiple patients
def generate_multiple_patients(n=1):
    """
    Genera n dizionari di pazienti casuali.

    Args:
        n (int): Numero di pazienti da generare

    Returns:
        list: Lista di dizionari, uno per ogni paziente
    """
    return [generate_random_patient() for _ in range(n)]




single_patient = generate_random_patient()
# dataframe for a single patient



# Per fare predizioni su nuovi dati
multiple_patients = generate_multiple_patients(n=50)

host = os.getenv("HOST")


url = f"{host}:9696/predict"
response = requests.post(url, json=single_patient).json()

print(response)
# print(single_patient)