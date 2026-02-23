# modules/doctor_profiles.py

import pandas as pd
import os

# Get absolute path of the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "doctor_profiles.csv")

# Load doctor profiles safely
doctor_df = pd.read_csv(DATA_PATH)

def get_all_doctors():
    return doctor_df.to_dict(orient="records")
