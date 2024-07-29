from abc import ABC, abstractmethod
import anndata as ad

class GeneHierarchy(ABC):
    def __init__(self):
        self.__pseudo_time = None
        self.__binarized_data = None
        self.__transition_points = None

    @abstractmethod
    def calculate_hierarchy(self, adata: ad.AnnData):
        """Calculate gene hierarchy based on the provided AnnData object."""
        pass

    @property
    def pseudo_time(self):
        return self.__pseudo_time

    @pseudo_time.setter
    def pseudo_time(self, value):
        self.__pseudo_time = value

    @property
    def binarized_data(self):
        return self.__binarized_data

    @binarized_data.setter
    def binarized_data(self, value):
        self.__binarized_data = value

    @property
    def transition_points(self):
        return self.__transition_points

    @transition_points.setter
    def transition_points(self, value):
        self.__transition_points = value

    @abstractmethod
    def _compute_transition_points(self):
        """Compute transition points between density regions."""
        pass