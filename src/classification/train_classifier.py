"""
Week 1 — Classification.

Week 1 — Classification.

Trains a per-pixel land-cover classifier (agricultural / built_up / water / barren)
on labeled samples with spectral bands, indices, and texture features.

Expects a training table (CSV or Parquet) with columns matching
src.config.FEATURE_COLUMNS plus a "label" column (see LAND_COVER_CLASSES).
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    FEATURE_COLUMNS,
    LAND_COVER_CLASSES,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    CLASSIFIER_TYPE,
    MODELS_DIR,
)


def load_training_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_model():
    if CLASSIFIER_TYPE == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(LAND_COVER_CLASSES),
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if CLASSIFIER_TYPE == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported CLASSIFIER_TYPE: {CLASSIFIER_TYPE}")


def train(training_table_path: Path, model_out_path: Path) -> None:
    df = load_training_table(training_table_path)
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Training table missing columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=list(LAND_COVER_CLASSES.values())))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out_path)
    print(f"Saved model to {model_out_path}")

    # Print feature importances
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nFeature importances (top 10):")
        for feat, imp in importances[:10]:
            print(f"  {feat:25s}: {imp:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the land-cover classifier.")
    parser.add_argument("training_table", type=Path, help="CSV/Parquet of labeled pixel samples.")
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / f"land_cover_{CLASSIFIER_TYPE}.joblib",
    )
    args = parser.parse_args()
    train(args.training_table, args.out)
