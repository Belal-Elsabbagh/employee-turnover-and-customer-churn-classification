from sklearn.ensemble import AdaBoostClassifier


def ada_boost(n_estimators=100):
    return AdaBoostClassifier(n_estimators=n_estimators)