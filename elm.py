import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

# ELM Script
class ELM:
    def __init__(self, data, activation, split_size):
        self.data = data
        self.activation = activation
        self.split_size = split_size

    def gaussian(self, x):
        return np.exp(-(x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def dataloader(self):
        self.features = np.array(self.data)[:, :-1]
        self.target = np.array(self.data)[:, -1].reshape(-1, 1)
        self.feature_shape = self.features.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=self.split_size, random_state=0
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def input_to_hidden(self, x, hidden_units):
        np.random.seed(0)
        Win = np.random.normal(size=[self.feature_shape, hidden_units]) * 0.01
        hidden_state = np.dot(x, Win)
        if self.activation == "gaussian":
            hidden_state = self.gaussian(hidden_state)
        elif self.activation == "tanh":
            hidden_state = self.tanh(hidden_state)
        elif self.activation == "sigmoid":
            hidden_state = self.sigmoid(hidden_state)
        return hidden_state

    def hidden_to_output(self, hidden_units):
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataloader()
        hidden_state = self.input_to_hidden(self.X_train, hidden_units)
        hidden_state_t = np.transpose(hidden_state)
        Wout = np.dot(
            np.linalg.inv(np.dot(hidden_state_t, hidden_state)),
            np.dot(hidden_state_t, self.y_train),
        )
        return Wout

    def loss(self, true, pred):
        return mse(true, pred, squared=False)

    def train(self):
        search_space = np.arange(1, 1000, 1)
        weights = [self.hidden_to_output(hidden_units) for hidden_units in search_space]
        mse_list = []
        for weight, hidden_units in zip(weights, search_space):
            hidden_state = self.input_to_hidden(self.X_train, hidden_units)
            pred = np.dot(hidden_state, weight)
            loss = self.loss(self.y_train, pred)
            mse_list.append(loss)
        min_idx = mse_list.index(min(mse_list))
        optimal_hidden_units = search_space[min_idx]
        optimal_weight = weights[min_idx]
        print(
            "Training RMSE : {}\nOptimal hidden units : {} ".format(
                mse_list[min_idx], optimal_hidden_units
            )
        )

        return optimal_hidden_units, optimal_weight

    def test(self, w, hidden_units):
        hidden_state = self.input_to_hidden(self.X_test, hidden_units)
        pred = np.dot(hidden_state, w)
        print("Test RMSE : {}".format(self.loss(self.y_test, pred)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ELM Module")
    parser.add_argument(
        "--activation",
        metavar="a",
        choices=["gaussian", "tanh", "sigmoid"],
        help="Activation function",
        required=True,
    )
    parser.add_argument(
        "--data_path", metavar="d", help="Path of the data files", required=True
    )
    parser.add_argument(
        "--split", metavar="s", help="Split size for test dataset", required=True
    )
    args = parser.parse_args()

    split_size = float(args.split)
    activation = args.activation
    data_path = args.data_path

    data_df = pd.read_csv(data_path)
    elm_obj = ELM(data_df, activation, split_size)
    hidden_units, weight = elm_obj.train()

    elm_obj.test(weight, hidden_units)
