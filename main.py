from preprocessing import load_datasets,preprocess_data,feature_selection,mutual_info_selection,scale_data,apply_pca
from train_model import train_models,evaluate_models,get_blinded_probabilities
def main():
    blinded_test_dataset, test_dataset, train_dataset = load_datasets()
    blinded_test_dataset, test_dataset, train_dataset, features, y_test, y_blinded_ID = preprocess_data(blinded_test_dataset, test_dataset, train_dataset)
    cleaned_features = feature_selection(features)
    X_selected = mutual_info_selection(cleaned_features, train_dataset)

    X_train = X_selected
    X_test = test_dataset[X_selected.columns]
    X_blind_test = blinded_test_dataset[X_selected.columns]

    X_train_scaled, X_test_scaled, X_blinded_scaled = scale_data(X_train, X_test, X_blind_test)
    X_train_pca, X_test_pca, X_blinded_pca = apply_pca(X_train_scaled, X_test_scaled, X_blinded_scaled)

    y_train = train_dataset["CLASS"]
    best_models = train_models(X_train_pca, y_train)
    evaluate_models(best_models, X_test_pca, y_test)
    probabilities_df = get_blinded_probabilities(best_models, X_blinded_pca, y_blinded_ID)

    probabilities_df.to_csv('blinded_test_class_probabilities_with_id.csv', index=False)


if __name__ == "__main__":
    main()