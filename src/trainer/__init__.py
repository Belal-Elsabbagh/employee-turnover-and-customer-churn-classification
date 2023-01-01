import datetime
from timeit import default_timer


def train_model(model, x_train, y_train):
    start = default_timer()
    model.fit(x_train, y_train)
    print(f'Trained model in {datetime.timedelta(seconds=(default_timer() - start))}')
    return model