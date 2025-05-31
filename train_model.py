import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score

def train_models(X_train_pca, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
    best_models = {}
    for model_name in models:
        grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='accuracy')
        grid_search.fit(X_train_pca, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best {model_name}: {grid_search.best_params_}")
    return best_models

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auroc = roc_auc_score(y, y_proba)
    sensitivity = recall_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y, y_pred)

    return {
        'Accuracy': accuracy,
        'AUROC': auroc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1-score': f1
    }

def evaluate_models(best_models, X_test_pca, y_test):
    metrics = {}
    for model_name, model in best_models.items():
        metrics[model_name] = evaluate_model(model, X_test_pca, y_test)
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)
    return metrics_df

def get_blinded_probabilities(best_models, X_blinded_pca, y_blinded_ID):
    probabilities = []
    for model_name, model in best_models.items():
        probas = model.predict_proba(X_blinded_pca)
        probas_df = pd.DataFrame(probas, columns=[f'{model_name}_Class_{i}' for i in range(probas.shape[1])])
        probabilities.append(probas_df)
    probabilities_df = pd.concat(probabilities, axis=1)
    probabilities_df['id'] = y_blinded_ID.values
    return probabilities_df