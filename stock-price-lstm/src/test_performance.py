import pandas as pd
import matplotlib.pyplot as plt

from src import TrainModel


class TestPerformance:
    def __init__(self, model_path):
        trained = TrainModel(model_path=model_path)
        trained.predict()

        predictions = trained.predictions
        actuals = trained.val_data.data
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

        fig, ax = plt.subplots(figsize=(8,8))
        ax.plot(overall_returns_hist, label="Returns (Model)")
        ax.plot(expected_returns, label="Expected Returns")
        plt.legend()
        plt.show()
        #     ld[largest.name] = largest.index.values
        #
        # highest_pred = pd.DataFrame(ld).transpose()
        #
        # for row in highest_pred

if __name__ == "__main__":
    TestPerformance(r"E:\University\GraphNeuralNetworkProject\subprojects\base-lstm\stock-price-lstm\models\model-lstm-27042020")