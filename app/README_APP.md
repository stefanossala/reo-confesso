# App A — Diabetes Risk (plain model)

Minimal Android app that runs the on-device model `model.tflite` (the overfit
target model) and shows a 5-year diabetes risk. The model sits unprotected in
`assets/` — this is the "easy to extract" baseline for phase 4. App B (obfuscated)
comes later.

## How to create the project in Android Studio

1. New Project -> **Empty Views Activity** (Kotlin).
2. Name: `Diabetes Risk`. Package name: **`com.reoconfesso.diabetesrisk`** (use exactly
   this, so the provided files line up). Minimum SDK: **API 26**.
3. Let Gradle finish syncing.

## Files to use (copy these over the generated ones)

- `src/main/java/com/reoconfesso/diabetesrisk/MainActivity.kt`
- `src/main/java/com/reoconfesso/diabetesrisk/DiabetesModel.kt`
- `src/main/res/layout/activity_main.xml`
- `src/main/assets/model.tflite`          (create the `assets` folder if missing)
- `src/main/assets/preprocessing.json`

In Android Studio the `assets` folder is created via:
right-click `app` -> New -> Folder -> Assets Folder.

## Gradle changes (app-level build.gradle.kts)

Inside the `android { }` block, stop Gradle from compressing the model:

    androidResources {
        noCompress += "tflite"
    }

Inside `dependencies { }`, add the TensorFlow Lite runtime:

    implementation("org.tensorflow:tensorflow-lite:2.14.0")

Then Sync, Run on an emulator, and test with a row from `data/target_train.csv`
(remember: identifiers like name/city are NOT entered — only the clinical fields).

## Note on feature order
The app reads `feature_order`, `mean` and `std` from `preprocessing.json`, so the
normalisation always matches the Python training side. Do not hand-edit those.
