from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import os
import sys

from help_functions import load_data


# result model and pca
def knn_pca(features, labels, neighbors=5, pca_components=300):
    pc = PCA(n_components=pca_components, whiten=True)
    features = pc.fit_transform(features)
    knn_model = KNeighborsClassifier(n_neighbors=neighbors).fit(features, labels)
    return knn_model, pc


def knn(features, labels, neighbors=5):
    return KNeighborsClassifier(n_neighbors=neighbors).fit(features, labels)


def svm_pca(features, labels, pca_components=300):
    pc = PCA(n_components=pca_components, whiten=True)
    features = pc.fit_transform(features)
    return OneVsRestClassifier(LinearSVC(max_iter=10000, dual=False)).fit(features, labels), pc


if __name__ == '__main__':
    n_neighbors = [1, 3, 5, 7, 13, 21]
    n_components = [100, 200, 300, 400, 500]
    if len(sys.argv) > 2:
        n_neighbors = [int(sys.argv[1])]
    if len(sys.argv) == 3:
        n_components = [int(sys.argv[2])]

    # load data we need
    path_test_features = os.path.join('DukeMTMC-reID', 'save_retinex', 'test_features.txt')
    path_query_features = os.path.join('DukeMTMC-reID', 'save_retinex', 'query_features.txt')
    test_features = load_data(path_test_features)
    query_features = load_data(path_query_features)

    # split data to features and labels
    X_train = test_features[:, 0:test_features.shape[1] - 2]
    y_train = test_features[:, test_features.shape[1] - 2]

    X_test = query_features[:, 0:query_features.shape[1] - 2]
    y_test = query_features[:, query_features.shape[1] - 2]

    # KNN
    for c in n_components:
        for n in n_neighbors:
            # train model
            model, pca = knn_pca(X_train, y_train, n, c)
            x = pca.transform(X_test)

            # get result
            score = model.score(x, y_test)
            print('KNN-{} PCA-{} score: {}'.format(n, c, score))

    # SVM
    for c in n_components:
        # train model
        model, pca = svm_pca(X_train, y_train, c)
        x = pca.transform(X_test)

        # get result
        score = model.score(x, y_test)
        print('SVM PCA-{} score: {}'.format(c, score))
