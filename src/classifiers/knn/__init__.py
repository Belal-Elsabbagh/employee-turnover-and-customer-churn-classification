from sklearn.neighbors import KNeighborsClassifier


def knn(k=3):
    return KNeighborsClassifier(n_neighbors=k)