import pandas as pd
import os

# Get absolute path of the CSV
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "disease_to_doctor.csv")

# Load CSV
disease_df = pd.read_csv(CSV_PATH)

def predict_specialist(disease_name):
    # Normalize input
    disease_name = disease_name.strip().lower()

    # Normalize CSV columns
    disease_df['Disease'] = disease_df['Disease'].str.strip().str.lower()
    disease_df['Specialist'] = disease_df['Specialist'].str.strip()

    # Match disease
    match = disease_df[disease_df['Disease'] == disease_name]

    if not match.empty:
        return match['Specialist'].values[0]
    else:
        return None
