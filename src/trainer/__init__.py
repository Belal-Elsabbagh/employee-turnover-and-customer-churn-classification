import datetime
from timeit import default_timer

from src.model_handler import save_model


def train_model(model, x_train, y_train, save_to):
    start = default_timer()
    model.fit(x_train, y_train)
    print(f'Trained model in {datetime.timedelta(seconds=(default_timer() - start))}\a')
    save_model(model, f'out\\{save_to}')
    return model