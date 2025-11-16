
import pandas as pd
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_data(csv_path):
    """
    Load data from a CSV file.
    """
    print(f"Loading data from {csv_path}...")
    return pd.read_csv(csv_path)

def preprocess_train(df):
    """
    Preprocess the training data.
    """
    print("Preprocessing training data...")
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y_encoded, label_encoder, scaler, imputer

def preprocess_test(df, label_encoder, scaler, imputer):
    """
    Preprocess the test data using fitted transformers.
    """
    print("Preprocessing test data...")
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]

    X_imputed = imputer.transform(X)
    y_encoded = label_encoder.transform(y)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled, y_encoded

def train_and_evaluate(X_train, X_test, y_train, y_test, save_models=False):
    """
    Train and evaluate multiple models.
    """
    models = {
        "SVM": SVC(kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {f1:.4f}")

        results.append({"Model": name, "Accuracy": accuracy, "F1-score": f1})

        if save_models:
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} model to {filename}")

    return pd.DataFrame(results)

def main():
    """
    Main function to run the training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate music genre classification models.")
    parser.add_argument("--csv_path", type=str, default="data/features_songs.csv",
                        help="Path to the feature dataset CSV.")
    parser.add_argument("--save_models", action="store_true",
                        help="Save the trained models to disk.")
    args = parser.parse_args()

    df = load_data(args.csv_path)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, 2])

    X_train, y_train, label_encoder, scaler, imputer = preprocess_train(train_df)
    X_test, y_test = preprocess_test(test_df, label_encoder, scaler, imputer)

    if args.save_models:
        joblib.dump(label_encoder, 'label_encoder.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(imputer, 'imputer.pkl')
        print("Saved label encoder, scaler, and imputer.")

    results_df = train_and_evaluate(X_train, X_test, y_train, y_test, args.save_models)

    print("--- Model Comparison ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
