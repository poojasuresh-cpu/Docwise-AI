import os
import pandas as pd

# get the absolute path to the CSV, no matter where you run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "doctor_profiles.csv")

doctor_df = pd.read_csv(CSV_PATH)

def get_doctors_by_specialist(specialist, location=None, min_experience=0, min_rating=None):
    # Ensure consistent formatting
    doctor_df['Specialist'] = doctor_df['Specialist'].str.strip().str.lower()
    specialist = specialist.strip().lower()

    filtered = doctor_df[doctor_df['Specialist'] == specialist]

    # Filter by location (optional)
    if location:
        filtered = filtered[filtered['Location'].str.strip().str.lower() == location.strip().lower()]

    # Filter by experience
    filtered = filtered[filtered['Experience'] >= min_experience]

    # Filter by rating if column exists
    if min_rating is not None and "Rating" in filtered.columns:
        filtered = filtered[filtered['Rating'] >= min_rating]
        # Sort by rating first, then experience
        filtered = filtered.sort_values(by=["Rating", "Experience"], ascending=[False, False])
    else:
        # Only sort by experience if rating is missing
        filtered = filtered.sort_values(by="Experience", ascending=False)

    return filtered