import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    X, y = load_iris(return_X_y=True)

    print("Training the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, "iris_model.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save()