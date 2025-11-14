import abc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class BenchmarkDataset(abc.ABC):
    """
    Abstract base class for benchmark datasets.
    All datasets should inherit from this class and implement its abstract methods.
    """
    def __init__(self, name):
        self.name = name
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.label_binarizer = None
        self.label_mapping = None

    @abc.abstractmethod
    def load_data(self):
        """
        Loads and preprocesses the dataset.
        This method should populate X_train, y_train, X_test, y_test,
        and label_mapping.
        """
        pass

    def get_dataset_info(self):
        """
        Returns information about the dataset.
        """
        return {
            "name": self.name,
            "X_train_shape": self.X_train.shape if self.X_train is not None else None,
            "y_train_shape": self.y_train.shape if self.y_train is not None else None,
            "X_test_shape": self.X_test.shape if self.X_test is not None else None,
            "y_test_shape": self.y_test.shape if self.y_test is not None else None,
            "num_classes": self.y_train.shape[1] if self.y_train is not None else None,
            "label_mapping": self.label_mapping
        }

class DatasetRegistry:
    """
    A registry for managing different benchmark datasets.
    """
    _datasets = {}

    @classmethod
    def register_dataset(cls, dataset_class):
        """
        Decorator to register a dataset class.
        """
        if not issubclass(dataset_class, BenchmarkDataset):
            raise ValueError("Registered class must inherit from BenchmarkDataset")
        cls._datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    @classmethod
    def get_dataset(cls, dataset_name):
        """
        Retrieves an instance of a registered dataset.
        """
        dataset_class = cls._datasets.get(dataset_name)
        if not dataset_class:
            raise ValueError(f"Dataset '{dataset_name}' not registered.")
        return dataset_class(dataset_name)

    @classmethod
    def get_available_datasets(cls):
        """
        Returns a list of names of all registered datasets.
        """
        return list(cls._datasets.keys())

# Example of how a dataset would be implemented and registered (SignMnistDataset will be next)
# @DatasetRegistry.register_dataset
# class ExampleDataset(BenchmarkDataset):
#     def __init__(self, name="ExampleDataset"):
#         super().__init__(name)
#
#     def load_data(self):
#         # Dummy data for example
#         self.X_train = np.random.rand(100, 28, 28, 1)
#         self.y_train = np.eye(100, 10) # 10 classes
#         self.X_test = np.random.rand(20, 28, 28, 1)
#         self.y_test = np.eye(20, 10)
#         self.label_mapping = {i: str(i) for i in range(10)}
#         print(f"{self.name} data loaded.")
