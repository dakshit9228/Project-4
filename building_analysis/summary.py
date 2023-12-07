import pandas as pd
import matplotlib.pyplot as plt

class BuildingDatasetSummary:
    """
    This class provides a summary of the building dataset. It includes methods to return the shape of the dataset,
    attributes information, and a descriptive analysis.

    Attributes:
    ----------
    building_dataset : DataFrame
        The building dataset loaded from a CSV file.

    Methods:
    -------
    dataset_shape()
        Returns the shape of the dataset.

    dataset_info()
        Prints the information about dataset attributes.

    dataset_description()
        Returns descriptive statistics of the dataset.
    """

    def __init__(self, dataset):
        """
        Initializes the BuildingDatasetSummary with the provided dataset.

        Parameters:
        ----------
        dataset : DataFrame
            The building dataset loaded from a CSV file.
        """
        self.building_dataset = dataset

    def dataset_shape(self):
        """
        Returns the shape of the building dataset.

        Returns:
        -------
        tuple
            A tuple representing the number of rows and columns in the dataset.
        """
        return self.building_dataset.shape

    def dataset_info(self):
        """
        Prints the information about dataset attributes, including the number of non-null entries and the data type of each column.
        """
        return self.building_dataset.info()

    def dataset_description(self):
        """
        Returns descriptive statistics of the dataset, including count, mean, std, min, 25%, 50%, 75%, and max for numeric columns.

        Returns:
        -------
        DataFrame
            A DataFrame containing descriptive statistics of the dataset.
        """
        return self.building_dataset.describe(include='all')

    def missing_values(self):
        """Returns the count of missing values in each column."""
        return self.building_dataset.isnull().sum()

    def unique_value_counts(self):
        """Returns the count of unique values for each column."""
        unique_counts = {col: self.building_dataset[col].nunique() for col in self.building_dataset.columns}
        return unique_counts

    def correlation_matrix(self):
        """Returns the correlation matrix for numeric columns."""
        return self.building_dataset.corr()

    def data_types(self):
        """Returns the data types of each column."""
        return self.building_dataset.dtypes

    def sample_data(self, n=5):
        """Returns a random sample of n rows from the dataset."""
        return self.building_dataset.sample(n)

    def column_value_frequencies(self, column_name):
        """
        Returns the frequency of each category in the specified column.

        Parameters:
        ----------
        column_name : str
            The name of the column for which to calculate value frequencies.

        Returns:
        -------
        Series
            A Series containing the counts of each unique value in the specified column.
        """
        if column_name in self.building_dataset.columns:
            return self.building_dataset[column_name].value_counts()
        else:
            return f"Column '{column_name}' not found in the dataset."

    def all_column_frequencies(self):
        """
        Returns the frequency of values for each categorical column in the dataset.

        Returns:
        -------
        dict
            A dictionary containing frequencies for each categorical column.
        """
        frequencies = {}
        for column in self.building_dataset.columns:
            if self.building_dataset[column].dtype == 'object':
                frequencies[column] = self.building_dataset[column].value_counts()
        return frequencies

    def basic_histogram(self, column_name):
        """
        Generates a basic histogram for a specified numeric column.

        Parameters:
        ----------
        column_name : str
            The name of the numeric column for which to generate the histogram.

        Returns:
        -------
        A histogram plot of the specified column.
        """
        if column_name in self.building_dataset.columns:
            self.building_dataset[column_name].hist()
        else:
            return f"Column '{column_name}' not found in the dataset or is not numeric."

    def plot_all_histograms(self):
        """
        Plots histograms for all numeric columns in the dataset.
        """
        numeric_columns = self.building_dataset.select_dtypes(include='number').columns
        for column in numeric_columns:
            self.building_dataset[column].hist()
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()


