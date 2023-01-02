import datetime
from timeit import default_timer


def train_model(model, x_train, y_train):
    start = default_timer()
    model.fit(x_train, y_train)
    print(f'Trained {model.__class__.__name__} model in {round(default_timer() - start, 2)} seconds.')
    return model