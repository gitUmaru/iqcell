from base_class import Binarizer
import numpy as np
import anndata as ad

class Mean(Binarizer):
    def __init__(self, threshold=None):
        super().__init__()
        self.__threshold = threshold

    @property
    def threshold(self):
        return self.__threshold

    def discretize(self, data):
        if not self._trained:
            print("Training Means Binarizer...")
            self._fit(data)
        discretized_values = self.__apply_threshold(data.X)
        
        return ad.AnnData(X=discretized_values, obs=data.obs, var=data.var, raw=data.raw)

    def _fit(self, data):
        X = data.X
        num_cols = X.shape[1]
        self.__threshold = np.zeros(num_cols)

        for col in range(num_cols):
            print(f"Calculating threshold for column {col + 1}/{num_cols}...")
            column_data = X[:, col]
            self.__threshold[col] = np.mean(column_data)
        
        self._trained = True

    def __apply_threshold(self, values):
        num_cols = values.shape[1]
        discretized = np.zeros_like(values)

        for col in range(num_cols):
            discretized[:, col] = np.where(values[:, col] >= self.__threshold[col], 1, 0)

        return discretized
