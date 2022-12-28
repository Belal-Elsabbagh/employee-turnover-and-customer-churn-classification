from sklearn.svm import SVC


def svm():
    return SVC(kernel='linear', gamma='scale', shrinking=False)