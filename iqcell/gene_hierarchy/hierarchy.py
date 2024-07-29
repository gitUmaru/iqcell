import numpy as np
from scipy.ndimage import uniform_filter1d
from base_class import GeneHierarchy
import anndata as ad

class IqCellGeneHierarchy(GeneHierarchy):
    def __init__(self, window_length=None):
        super().__init__()
        self.__window_length = window_length

    @property
    def window_length(self):
        return self.__window_length

    @window_length.setter
    def window_length(self, value):
        self.__window_length = value

    def calculate_hierarchy(self, adata: ad.AnnData):
        self.binarized_data = adata.X
        self.__gene_names = adata.var_names

        num_genes = self.binarized_data.shape[1]

        if self.window_length is None:
            self.window_length = len(self.pseudo_time) // num_genes

        print("Computing density representation...")
        self._compute_density_representation()

        print("Computing transition points...")
        self._compute_transition_points()

        return self.transition_points

    def _compute_density_representation(self):
        smoothed_data = np.zeros_like(self.binarized_data)
        for gene_idx in range(self.binarized_data.shape[1]):
            smoothed_data[:, gene_idx] = uniform_filter1d(
                self.binarized_data[:, gene_idx], size=self.window_length, mode='reflect'
            )
        self.binarized_data = smoothed_data

    def _compute_transition_points(self):
        transition_points = []
        for gene_idx in range(self.binarized_data.shape[1]):
            data = self.binarized_data[:, gene_idx]
            transition_point = np.argmax(data >= 0.5) 
            
            gene_name = self.__gene_names[gene_idx]
            transition_points.append((gene_name, transition_point))

        self.transition_points = transition_points