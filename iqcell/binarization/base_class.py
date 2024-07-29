from abc import ABC, abstractmethod

class Binarizer(ABC):
    def __init__(self):
        self.__trained = False

    @property
    @abstractmethod
    def threshold(self):
        pass

    @abstractmethod
    def discretize(self, data):
        pass

    @abstractmethod
    def _fit(self, data):
        pass

    @property
    def _trained(self):
        return self.__trained

    @_trained.setter
    def _trained(self, value):
        self.__trained = value
