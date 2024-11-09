import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.feature_extraction import DictVectorizer

data_url = "https://www.kaggle.com/api/v1/datasets/download/rabieelkharoua/chronic-kidney-disease-dataset-analysis/Chronic_Kidney_Dsease_data.csv"
df = pd.read_csv(data_url)

df.drop(["DoctorInCharge", "PatientID"], axis=1, inplace=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.Diagnosis.values
y_val = df_val.Diagnosis.values
y_test = df_test.Diagnosis.values

del df_train["Diagnosis"]
del df_val["Diagnosis"]
del df_test["Diagnosis"]

dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.fit_transform(test_dict)

# Check class distribution to find if the dataset is unbalanced
def check_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print("Distribuzione delle classi:")
    for k, v in dist.items():
        print(f"Class {k}: {v} ({v/len(y)*100:.2f}%)")
    return dist

# XGBoost configuration
def create_balanced_xgboost():
    return XGBClassifier(
        n_estimators=100,  
        max_depth=3,       
        learning_rate=0.1, 
        objective='binary:logistic',
        scale_pos_weight=1, # to correct unbalanced datasets
        random_state=42
    )

# Training and monitoring
def train_and_evaluate_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    dist = check_class_distribution(y_train)
    if len(dist) == 2:
        neg, pos = dist[0], dist[1]
        scale_pos_weight = neg/pos
    else:
        scale_pos_weight = 1

    # Create and configure model
    model = create_balanced_xgboost()
    model.set_params(scale_pos_weight=scale_pos_weight)

    # Parameters for grid search
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 200],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Training with early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )

    # Grid Search to find best parameters
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    # Fit the best model
    best_model = grid_search.fit(X_train, y_train).best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)

    print("\nReport di classificazione:")
    print(classification_report(y_test, y_pred))

    print("\nMatrice di confusione:")
    print(confusion_matrix(y_test, y_pred))

    return best_model

# Return predictions
def verify_predictions(model, X):
    preds = model.predict(X)
    pred_probs = model.predict_proba(X)

    print("\nPredictions stats:")
    print(f"Unique predictions: {np.unique(preds, return_counts=True)}")
    print("\n")
    print(f"Min prob class 1: {pred_probs[:, 1].min():.3f}")
    print(f"Max prob class 1: {pred_probs[:, 1].max():.3f}")
    print(f"Average prob class 1: {pred_probs[:, 1].mean():.3f}")

    return preds, pred_probs

# Main function
def main():

    # Training and evaluation
    best_model = train_and_evaluate_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    # Verify predictions
    verify_predictions(best_model, X_test)

    return best_model

# Create and train model
best_model = main()

# Save the model

save_file_name = "xgbm_final_model_local.pkl"

with open(save_file_name, 'wb') as file:
    pickle.dump((dv, best_model), file)

# Load the model
with open(save_file_name, 'rb') as file:
    dv, loaded_model = pickle.load(file)

