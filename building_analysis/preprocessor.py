import pandas as pd

class BuildingDatasetPreprocessor:
    """
    BuildingDatasetPreprocessor - A library for preprocessing building-related datasets.

    This library provides a class, BuildingDatasetPreprocessor, designed for efficiently preprocessing datasets
    related to building information. It includes methods for checking and transforming data, ensuring the dataset
    adheres to specific requirements.

    Attributes:
        dataset (pd.DataFrame): The input dataset to be processed.

    Methods:
        __init__(self, dataset):
            Initializes the BuildingDatasetPreprocessor with the provided dataset.

        preprocess_data(self):
            Performs data preprocessing on the dataset, including:
                - Checking for the existence of required columns.
                - Converting 'Construction Date' to datetime format.
                - Calculating 'Building Age' based on the construction date.
                - Dropping rows with missing values.

            Returns:
                pd.DataFrame: The preprocessed dataset.

            Raises:
                ValueError: If required columns are missing in the dataset.

    Usage:
        # Instantiate the preprocessor with a dataset
        preprocessor = BuildingDatasetPreprocessor(my_dataset)

        # Preprocess the data
        preprocessed_data = preprocessor.preprocess_data()

    Author:
        Your Name

    Version:
        1.0.0
    """
    def __init__(self, dataset):
        """
        Initializes the BuildingDatasetPreprocessor with the provided dataset.

        Args:
            dataset (pd.DataFrame): The input dataset to be processed.
        """
        self.dataset = dataset.copy()

    def preprocess_data(self):
        """
        Perform data preprocessing on the dataset.

        Returns:
            pd.DataFrame: The preprocessed dataset.

        Raises:
            ValueError: If required columns are missing in the dataset.
        """
        # Ensure that the required columns exist
        required_columns = ['Construction Date']
        missing_columns = [col for col in required_columns if col not in self.dataset.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

        # Convert 'Construction Date' to datetime format
        self.dataset['Construction Date'] = pd.to_datetime(self.dataset['Construction Date'], errors='coerce')

        # Calculate 'Building Age' based on the construction date
        self.dataset['Building Age'] = 2023 - self.dataset['Construction Date'].dt.year

        # Drop rows with missing values
        self.dataset.dropna(inplace=True)

        return self.dataset
