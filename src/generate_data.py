"""Generate the synthetic diabetes-risk dataset described in docs/dataset_spec.md.

All records are synthetic. Running the script twice yields identical files (fixed seed).

Usage (from the repository root):
    python src/generate_data.py
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers.address.it_IT import Provider as ItAddressProvider

SEED = 42
N = 6000
SPLIT_SIZE = 2000
BASE_SCORE = -2.2  # calibrated: gives ~24% prevalence (spec target 15-25%)
REFERENCE_DATE = date(2026, 10, 1)  # "today" for computing dates of birth

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

rng = np.random.default_rng(SEED)
fake = Faker("it_IT")
fake.seed_instance(SEED)
# fake.city() is NOT reproducible across runs (its city collection depends on Python's
# hash seed), so we sort Faker's Italian city list once and sample from it with numpy.
CITIES = sorted(ItAddressProvider.cities)

# ---------------------------------------------------------------------------
# 1. Independent features
# ---------------------------------------------------------------------------
age = rng.integers(18, 81, size=N)                                   # 18-80 inclusive
sex = rng.choice(["M", "F"], size=N)                                 # 50/50
bmi = np.clip(rng.normal(27, 5, size=N), 16, 50)
activity_h = np.clip(rng.exponential(3, size=N), 0, 15)              # hours per week
smoker = rng.random(N) < 0.20
family_history = rng.random(N) < 0.30
zone = rng.choice(list("ABCDE"), size=N)                             # decoy: no effect on label

# ---------------------------------------------------------------------------
# 2. Dependent features (computed after age and BMI)
# ---------------------------------------------------------------------------
systolic_bp = np.clip(
    105 + 0.4 * age + 0.8 * (bmi - 25) + rng.normal(0, 12, size=N), 90, 200
).round()
glucose = np.clip(
    90 + 0.3 * (age - 40) + 1.2 * (bmi - 25) + rng.normal(0, 10, size=N), 65, 200
)

# ---------------------------------------------------------------------------
# 3. Label: risk score -> probability -> random draw (intended noise)
# ---------------------------------------------------------------------------
score = (
    BASE_SCORE
    + 0.04 * (age - 45)
    + 0.12 * (bmi - 25)
    + 0.05 * (glucose - 95)
    + 0.8 * family_history
    + 0.015 * (systolic_bp - 120)
    - 0.10 * activity_h
    + 0.3 * smoker
    + 0.1 * (sex == "M")
)
prob = 1 / (1 + np.exp(-score))
onset_5y = (rng.random(N) < prob).astype(int)

# ---------------------------------------------------------------------------
# 4. Identifiers (NOT model features)
# ---------------------------------------------------------------------------
names, cities, birth_dates = [], [], []
for i in range(N):
    first = fake.first_name_male() if sex[i] == "M" else fake.first_name_female()
    names.append(f"{first} {fake.last_name()}")  # no titles such as "Dott."
    cities.append(CITIES[int(rng.integers(len(CITIES)))])
    # Date of birth consistent with age on REFERENCE_DATE
    days_since_birthday = int(rng.integers(0, 365))
    birth_year = REFERENCE_DATE.year - int(age[i])
    birthday_this_age = REFERENCE_DATE - timedelta(days=days_since_birthday)
    try:
        dob = birthday_this_age.replace(year=birthday_this_age.year - int(age[i]))
    except ValueError:  # 29 February in a non-leap year
        dob = date(birth_year, 2, 28)
    birth_dates.append(dob.isoformat())

df = pd.DataFrame(
    {
        "patient_id": [f"P{i:05d}" for i in range(N)],
        "name": names,
        "date_of_birth": birth_dates,
        "city": cities,
        "age": age,
        "sex": sex,
        "bmi": bmi.round(1),
        "activity_h": activity_h.round(1),
        "smoker": smoker.astype(int),
        "family_history": family_history.astype(int),
        "systolic_bp": systolic_bp.astype(int),
        "glucose": glucose.round(1),
        "zone": zone,
        "onset_5y": onset_5y,
    }
)

# ---------------------------------------------------------------------------
# 5. Sanity checks
# ---------------------------------------------------------------------------
prevalence = df["onset_5y"].mean()
print(f"Records: {len(df)}")
print(f"Prevalence of onset_5y: {prevalence:.1%}  (target 15-25%)")
print(df[["age", "bmi", "activity_h", "systolic_bp", "glucose"]].describe().round(1))
if not 0.15 <= prevalence <= 0.25:
    print("WARNING: prevalence outside target range -> adjust BASE_SCORE")

# ---------------------------------------------------------------------------
# 6. Shuffle, split, save
# ---------------------------------------------------------------------------
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
splits = {
    "target_train": df.iloc[:SPLIT_SIZE],                   # members
    "target_out": df.iloc[SPLIT_SIZE:2 * SPLIT_SIZE],       # non-members
    "shadow_pool": df.iloc[2 * SPLIT_SIZE:3 * SPLIT_SIZE],  # attacker's data
}
DATA_DIR.mkdir(exist_ok=True)
for split_name, part in splits.items():
    path = DATA_DIR / f"{split_name}.csv"
    part.to_csv(path, index=False)
    print(f"Saved {path.name}: {len(part)} rows, prevalence {part['onset_5y'].mean():.1%}")
