from base_class import Binarizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import anndata as ad

class KMeans(Binarizer):
    def __init__(self, threshold=None):
        super().__init__()
        self.__threshold = threshold
        self.__kmeans = None

    @property
    def threshold(self):
        return self.__threshold

    def discretize(self, data):
        if not self._trained:
            print("Training K-Means model...")
            self._fit(data)
        discretized_values = self.__apply_threshold(data.X)

        return ad.AnnData(X=discretized_values, obs=data.obs, var=data.var, raw=data.raw)

    def _fit(self, data):
        X = data.raw.X
        num_cols = X.shape[1]
        self.__threshold = np.zeros(num_cols)
        kmeans = KMeans(n_clusters=2)

        for col in tqdm(range(num_cols)):
            column_data = X[:, col].reshape(-1, 1)
            kmeans.fit(column_data)
            self.__threshold[col] = self.__determine_threshold(kmeans)
        
        self._trained = True

    def __determine_threshold(self, model):
        # Calculate the threshold as the mean of the centroids for the column
        centroids = model.cluster_centers_
        return np.mean(centroids)

    def __apply_threshold(self, values):
        num_cols = values.shape[1]
        discretized = np.zeros_like(values)

        for col in range(num_cols):
            discretized[:, col] = np.where(values[:, col] >= self.__threshold[col], 1, 0)

        return discretized
