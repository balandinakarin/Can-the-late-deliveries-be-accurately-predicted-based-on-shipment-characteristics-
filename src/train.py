import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = "data/Train.csv"
TARGET_COL = "Reached.on.Time_Y.N"


def print_top_features(clf, top_n=10):
# Srongest features for predicting:
# 1 (on time)
# 0 (late)

# For Logistic Regression:
# Positive coefficient - pushes prediction toward 1 (on time)
# Negative coefficient - pushes prediction toward 0 (late)

    # Preprocessing step from the pipeline
    preprocessor = clf.named_steps["preprocessor"]

    # Feature names AFTER OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Coefficients from Logistic Regression
    model = clf.named_steps["model"]
    coefs = model.coef_[0]  # one row because it's binary classification

    # Put into a table for easy sorting
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    }).sort_values(by="coefficient", ascending=False)

    # Print the top features for class 1 (on time)
    print("\n___TOP FEATURES PUSHING TOWARD ON-TIME (1)___")
    print(coef_df.head(top_n).to_string(index=False))

    # Print the top features for class 0 (late)
    print("\n___TOP FEATURES PUSHING TOWARD LATE (0)___")
    print(coef_df.tail(top_n).sort_values(by="coefficient").to_string(index=False))

    # 7) Tiny legend for interpretation
    print("\nHow to read this:")
    print("\nPositive coefficient - increases chance of ON-TIME (1)")
    print("\nNegative coefficient - increases chance of LATE (0)")


def main():
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      # 80% training, 20% testing
        random_state=42,    # same split each run
        stratify=y          # keep similar 0/1 proportions in train/test
    )

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing:
    # OneHotEncode text columns
    # Keep numeric columns unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Logistic Regression model
    model = LogisticRegression(max_iter=2000)

    # Combine preprocessing + model
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Print top features
    print_top_features(clf, top_n=10)


if __name__ == "__main__":
    main()
