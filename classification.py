from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import numpy as np
import os

import help_functions


# result model and pca
def knn_pca(features, labels, neighbors=5, pca_components=300):
    pc = PCA(n_components=pca_components, whiten=True)
    features = pca.fit_transform(features)
    knn_model = KNeighborsClassifier(n_neighbors=neighbors).fit(features, labels)
    return knn_model, pc


def knn(features, labels, neighbors=5):
    return KNeighborsClassifier(n_neighbors=neighbors).fit(features, labels)


if False:
    # load data we need
    path_test_features = os.path.join('DukeMTMC-reID', 'test_features.txt')
    path_query_features = os.path.join('DukeMTMC-reID', 'test_features.txt')
    test_features = help_functions.load_data(path_test_features)
    query_features = help_functions.load_data(path_query_features)

    # split data to features and labels
    X_train_knn = test_features[:, 0:test_features.shape[1] - 2]
    y_train_knn = test_features[:, test_features.shape[1] - 2]

    X_test_knn = query_features[:, 0:query_features.shape[1] - 2]
    y_test_knn = query_features[:, query_features.shape[1] - 2]

    # train model
    model, pca = knn_pca(X_train_knn, y_train_knn)
    pca.transform(X_test_knn)

    # get result
    score = model.score(X_test_knn, y_test_knn)
    print('KNN score: ' + score)
