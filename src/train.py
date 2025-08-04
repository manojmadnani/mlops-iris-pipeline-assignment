# src/train.py
from mlflow.models.signature import infer_signature
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from utils import load_data

df = load_data()
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=200)
}

mlflow.set_experiment("iris_experiment")


for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Prepare example input and signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name="model", input_example=input_example, signature=signature)
        joblib.dump(model, f"models/{name}.pkl")
