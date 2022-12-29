from sklearn.svm import SVC


def svm():
    return SVC(class_weight="balanced")