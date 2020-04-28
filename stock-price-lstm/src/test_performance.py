import pandas as pd
import matplotlib.pyplot as plt

from src import TrainModel


class TestPerformance:
    def __init__(self, model_path):
        trained = TrainModel(model_path=model_path)

        predictions = trained.predictions
        actuals = trained.val_data.data
        actuals = actuals.rolling(7).sum()
        actuals = actuals[
            (actuals.index >= predictions.index[0]) & (actuals.index <= predictions.index[-1])
            ]

        ld = {}

        n = 5
        overall_returns_hist = []
        overall_returns = 0
        for row in predictions.iterrows():
            daily_returns = 0
            largest = row[1].nlargest(n)
            for i in range(n):
                daily_returns += actuals[largest.index.values[i]][largest.name] / n
            overall_returns += daily_returns
            overall_returns_hist.append(overall_returns)

        expected_returns = actuals.mean(axis=1).cumsum().values

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(overall_returns_hist, label="Returns (Model)")
        ax.plot(expected_returns, label="Expected Returns")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("load_model", help="Location of saved model state.")
    args = parser.parse_args()

    TestPerformance(args.load_model)
