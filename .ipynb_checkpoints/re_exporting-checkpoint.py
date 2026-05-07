# reexport_models.py
import joblib

paths = [
    "bayesian/naive_bayes_gridsearch.pkl",
    "Gradient_Boosting/gradient_boosting_gridsearch.pkl",
    "KNN/knn_cardio_pipeline.pkl",
    "KNN/knn_cardio_model.pkl",
    "Logistic_Regression/logistic_regression.pkl",
    "SVM/svm_gridsearch.pkl",
    "Random_Forest/random_forest_gridsearch.pkl",
]

for path in paths:
    try:
        model = joblib.load(path)
        joblib.dump(model, path, compress=0)   # compress=0 = protocolo más compatible
        print(f"OK: {path}")
    except Exception as e:
        print(f"ERROR {path}: {e}")