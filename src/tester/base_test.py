from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, f1_score

from src.dataset_handler import default_preprocess, load_csv_dataset
from src.trainer import train_model


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    print('F1: %.3f' % f1_score(y_test, y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))


def base_test(
        csv_path,
        base_model,
        index_col,
        target_col,
        test_size=0.2,
        exclude_cols=None,
        preprocess: callable = default_preprocess
):
    if exclude_cols is None:
        exclude_cols = []
    feature_train, feature_test, label_train, label_test = load_csv_dataset(
        csv_path, index_col, target_col,
        exclude_cols=exclude_cols, test_size=test_size, preprocess=preprocess)
    label_train = label_train[target_col].tolist()
    label_test = label_test[target_col].tolist()
    model = train_model(base_model, feature_train, label_train, 'model.sav')
    test_model(model, feature_test, label_test)
    print('Tested model\a')
