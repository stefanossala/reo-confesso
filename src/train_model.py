"""Train the target models on the synthetic diabetes dataset and export them to TensorFlow Lite.

Two target models are trained on the SAME members (data/target_train.csv):
  - "normal":  small regularised network, epochs chosen by early stopping
  - "overfit": larger network, no regularisation, many epochs (memorises its training set)

Methodological note: the validation split used for early stopping is NOT used for the final
model. We first find the best number of epochs on an 80/20 split, then retrain from scratch on
all 2,000 members. This way, every record in target_train.csv is a true member of the final
model, which phase 2 (membership inference) relies on.

Usage (from the repository root):
    python src/train_model.py
"""
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

try:  # LiteRT is the successor of tf.lite.Interpreter
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
    Interpreter = tf.lite.Interpreter

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

LABEL = "onset_5y"
NUMERIC = ["age", "bmi", "activity_h", "systolic_bp", "glucose"]
BINARY = ["smoker", "family_history"]
ZONES = list("ABCDE")
# Fixed column order: the Android app must build its input vector in exactly this order.
FEATURE_ORDER = NUMERIC + ["sex_M"] + BINARY + [f"zone_{z}" for z in ZONES]

CONFIGS = {
    "normal": {"hidden": [64, 64], "l2": 1e-3, "dropout": 0.2, "max_epochs": 200,
               "batch_size": 32, "early_stopping": True},
    "overfit": {"hidden": [256, 256], "l2": 0.0, "dropout": 0.0, "max_epochs": 300,
                "batch_size": 32, "early_stopping": False},
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model features (identifiers are dropped) and encode categoricals."""
    out = df[NUMERIC + BINARY].astype("float32").copy()
    out["sex_M"] = (df["sex"] == "M").astype("float32")
    for z in ZONES:
        out[f"zone_{z}"] = (df["zone"] == z).astype("float32")
    return out[FEATURE_ORDER]


def fit_scaler(x: pd.DataFrame) -> dict:
    """Mean/std computed on the members only (target_train)."""
    mean = x.mean()
    std = x.std(ddof=0).replace(0, 1.0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def transform(x: pd.DataFrame, scaler: dict) -> np.ndarray:
    return ((x.values - np.array(scaler["mean"])) / np.array(scaler["std"])).astype("float32")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(n_features: int, cfg: dict) -> keras.Model:
    reg = keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for units in cfg["hidden"]:
        x = keras.layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if cfg["dropout"] > 0:
            x = keras.layers.Dropout(cfg["dropout"])(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="risk")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def choose_epochs(x: np.ndarray, y: np.ndarray, cfg: dict) -> int:
    """Pick the number of epochs on an 80/20 split of the members (validation only)."""
    if not cfg["early_stopping"]:
        return cfg["max_epochs"]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    keras.utils.set_random_seed(SEED)
    model = build_model(x.shape[1], cfg)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=cfg["max_epochs"],
                     batch_size=cfg["batch_size"], callbacks=[stop], verbose=0)
    return int(np.argmin(hist.history["val_loss"])) + 1


def evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray) -> dict:
    p = model.predict(x, verbose=0).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, p >= 0.5)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "loss": round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), 4),
    }


# ---------------------------------------------------------------------------
# TFLite export + parity check
# ---------------------------------------------------------------------------
def export_tflite(model: keras.Model, path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    path.write_bytes(converter.convert())


def check_tflite(model: keras.Model, path: Path, x: np.ndarray) -> float:
    """Max absolute difference between Keras and TFLite outputs on the given rows."""
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    tfl = []
    for row in x:
        interpreter.set_tensor(inp["index"], row.reshape(1, -1))
        interpreter.invoke()
        tfl.append(interpreter.get_tensor(out["index"]).ravel()[0])
    keras_out = model.predict(x, verbose=0).ravel()
    return float(np.max(np.abs(np.array(tfl) - keras_out)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    train_df = pd.read_csv(DATA_DIR / "target_train.csv")
    out_df = pd.read_csv(DATA_DIR / "target_out.csv")

    scaler = fit_scaler(to_feature_frame(train_df))
    x_train = transform(to_feature_frame(train_df), scaler)
    x_out = transform(to_feature_frame(out_df), scaler)
    y_train = train_df[LABEL].values
    y_out = out_df[LABEL].values

    preprocessing = {
        "feature_order": FEATURE_ORDER,
        "mean": scaler["mean"],
        "std": scaler["std"],
        "encoding": {"sex_M": "1 if sex == 'M' else 0",
                     "zone_X": "one-hot over zones A-E"},
        "output": "probability of type 2 diabetes onset within 5 years (sigmoid)",
        "threshold": 0.5,
    }
    (MODELS_DIR / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))

    all_metrics = {}
    for name, cfg in CONFIGS.items():
        print(f"\n=== Model '{name}' ===")
        epochs = choose_epochs(x_train, y_train, cfg)
        print(f"Epochs: {epochs}" + (" (chosen by early stopping)" if cfg["early_stopping"] else " (fixed)"))

        # Final model: retrain from scratch on ALL members
        keras.utils.set_random_seed(SEED)
        model = build_model(x_train.shape[1], cfg)
        model.fit(x_train, y_train, epochs=epochs, batch_size=cfg["batch_size"], verbose=0)

        m_train = evaluate(model, x_train, y_train)
        m_out = evaluate(model, x_out, y_out)
        gap = {k: round(m_train[k] - m_out[k], 4) for k in ("accuracy", "auc")}
        print(f"  members     (target_train): {m_train}")
        print(f"  non-members (target_out):   {m_out}")
        print(f"  generalization gap:         {gap}")

        model.save(MODELS_DIR / f"target_{name}.keras")
        tflite_path = MODELS_DIR / f"target_{name}.tflite"
        export_tflite(model, tflite_path)
        diff = check_tflite(model, tflite_path, x_out[:20])
        status = "OK" if diff < 1e-4 else "MISMATCH - check the conversion"
        print(f"  TFLite: {tflite_path.name} ({tflite_path.stat().st_size / 1024:.0f} KB), "
              f"max diff vs Keras = {diff:.2e} [{status}]")

        all_metrics[name] = {"config": cfg, "epochs": epochs, "members": m_train,
                             "non_members": m_out, "gap": gap, "tflite_max_diff": diff}

    baseline = round(float(max(y_out.mean(), 1 - y_out.mean())), 4)
    all_metrics["majority_class_accuracy"] = baseline
    print(f"\nReference: always predicting 'no onset' gives accuracy {baseline} on target_out.")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print("Saved models/preprocessing.json and models/metrics.json")


if __name__ == "__main__":
    main()
