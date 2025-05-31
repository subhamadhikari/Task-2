import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_datasets():
    blinded_test_dataset = pd.read_csv("blinded_test_set.csv")
    test_dataset = pd.read_csv("test_set.csv")
    train_dataset = pd.read_csv("train_set.csv")
    return blinded_test_dataset, test_dataset, train_dataset

def preprocess_data(blinded_test_dataset, test_dataset, train_dataset):
    columns = train_dataset.columns

    redundant_columns = []
    # zero variance features extraction
    for col in columns:
        if len(set(train_dataset[col])) == 1:
            redundant_columns.append(col)

    y_test = test_dataset["CLASS"]
    y_blinded_ID = blinded_test_dataset["ID"]

    blinded_test_dataset.drop("ID",axis=1,inplace=True)
    blinded_test_dataset.drop(redundant_columns,axis=1,inplace=True)
    redundant_columns.append("ID")
    redundant_columns.append("CLASS")

    features = train_dataset.drop(redundant_columns,axis=1)
    test_dataset = test_dataset.drop(redundant_columns,axis=1)
    
    # remove features with nan values
    cols_with_nan = features.columns[features.isnull().any()]
    features= features.drop(cols_with_nan,axis=1)
    test_dataset = test_dataset.drop(cols_with_nan,axis=1)
    blinded_test_dataset = blinded_test_dataset.drop(cols_with_nan,axis=1)

    # replave inifinitive value with median in train data
    cols_with_inf = features.columns[np.isinf(features).any()]
    print(cols_with_inf)
    features['Feature_72'].replace([np.inf, -np.inf], np.nan, inplace=True)
    features['Feature_72'].fillna(features['Feature_72'].median(), inplace=True)

    # remove highly correlated features
    correlation_matrix = features.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

    blinded_test_dataset.drop(to_drop, axis=1, inplace=True)
    features = features.drop(to_drop, axis=1)
    test_dataset = test_dataset.drop(to_drop, axis=1)

    return blinded_test_dataset, test_dataset, train_dataset, features, y_test, y_blinded_ID

def feature_selection(features):
    # remove features with variance less than 0.01
    selector = VarianceThreshold(threshold=0.01)
    features_var = selector.fit_transform(features)
    selected_columns = features.columns[selector.get_support()]
    cleaned_features = features[selected_columns]
    return cleaned_features

def mutual_info_selection(cleaned_features, train_dataset):
    mi = mutual_info_classif(cleaned_features, train_dataset["CLASS"])
    mi_series = pd.Series(mi, index=cleaned_features.columns)
    greater_mi = mi_series[mi_series > 0].index
    X_selected = cleaned_features[greater_mi]
    return X_selected

def scale_data(X_train, X_test, X_blind_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_blinded_scaled = scaler.transform(X_blind_test)
    return X_train_scaled, X_test_scaled, X_blinded_scaled

def apply_pca(X_train_scaled, X_test_scaled, X_blinded_scaled):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.show()

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_blinded_pca = pca.transform(X_blinded_scaled)
    return X_train_pca, X_test_pca, X_blinded_pca




