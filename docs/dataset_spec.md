# Synthetic dataset specification

**Scenario.** A (fictional) diabetes prevention programme trains an on-device model that estimates a user's
risk of developing type 2 diabetes within 5 years. The model ships inside a wellness app.

**Why this domain.** Health data is a special category under GDPR Art. 9, and a diagnostic-support model may
qualify as a medical device, hence high-risk under AI Act Art. 6(1). Above all, **membership itself is
sensitive**: learning that a person was in the training set reveals that they were a programme participant.

All records are synthetic. Any resemblance to real persons is coincidental.

## Identifiers (NOT model features)
- `patient_id` (unique), full name, date of birth consistent with age, city
- Generated with Faker (`it_IT` locale), fixed seed

## Features
| Feature | Type | Generation | Notes |
| --- | --- | --- | --- |
| `age` | int | uniform 18–80 | |
| `sex` | cat M/F | 50/50 | |
| `bmi` | float | normal, mean 27, sd 5, clipped 16–50 | |
| `activity_h` | float | exponential, mean 3 h/week, clipped 0–15 | |
| `smoker` | bool | 20% yes | |
| `family_history` | bool | 30% yes | |
| `systolic_bp` | int | 105 + 0.4·age + 0.8·(bmi − 25) + noise (sd 12), clipped 90–200 | depends on age, BMI |
| `glucose` | float | 90 + 0.3·(age − 40) + 1.2·(bmi − 25) + noise (sd 10), clipped 65–200 mg/dL | fasting; depends on age, BMI |
| `zone` | cat A–E | uniform | **no effect on label** (decoy feature, AI Act Art. 10 discussion) |

Generation order matters: age and BMI first, then the features that depend on them.

## Label rule (ground truth, unknown to the model)
`onset_5y` = 1 if the person develops type 2 diabetes within 5 years.

Risk score starts at **−2.2** (calibrated; first draft was −1.9, which gave 28%), then:

| Factor | Effect on score |
| --- | --- |
| Age | +0.04 per year above 45 (negative below) |
| BMI | +0.12 per point above 25 |
| Fasting glucose | +0.05 per mg/dL above 95 |
| Family history | +0.8 |
| Systolic BP | +0.015 per mmHg above 120 |
| Physical activity | −0.10 per weekly hour |
| Smoker | +0.3 |
| Male | +0.1 |

Probability = 1 / (1 + exp(−score)).
Label = 1 if a uniform random draw < probability (intended noise: identical people can have different outcomes).

**Target prevalence: 15–25%.** If outside, adjust only the base score (higher → more cases, lower → fewer).

## Size and split
- Total: 6,000 records, fixed random seed (42)
- `target_train.csv`: 2,000 records (members, used to train the target model)
- `target_out.csv`: 2,000 records (non-members, never seen by the target model)
- `shadow_pool.csv`: 2,000 records (for the attacker's shadow models)
- Hyperparameter tuning uses a validation split carved from `target_train` only.
