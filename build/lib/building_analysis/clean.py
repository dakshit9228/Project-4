import numpy as np
import pandas as pd


class BuildingDatasetCleaner:
    """
    This class is designed for cleaning the building dataset.

    Methods:
    -------
    fill_missing_values(column_name, method='mean')
        Fills missing values in a specified column using a defined method ('mean', 'median', or 'mode').

    drop_missing_values()
        Drops rows with any missing values.

    convert_to_datetime(column_name, format_string='%d-%b-%Y')
        Converts a specified column to a datetime format.
    """

    def __init__(self, dataset):
        """
        Initializes the BuildingDatasetCleaner with the provided dataset.

        Parameters:
        ----------
        dataset : DataFrame
            The building dataset loaded from a CSV file.
        """
        self.building_dataset = dataset

    def fill_missing_values(self, column_name, method="mean"):
        """
        Fills missing values in a specified column using a defined method ('mean', 'median', or 'mode').

        Parameters:
        ----------
        column_name : str
            The name of the column to fill missing values in.
        method : str, optional
            The method to use for filling missing values ('mean', 'median', 'mode'). Default is 'mean'.
        """
        if column_name not in self.building_dataset.columns:
            return f"Column '{column_name}' not found in the dataset."

        if method == "mean":
            fill_value = self.building_dataset[column_name].mean()
        elif method == "median":
            fill_value = self.building_dataset[column_name].median()
        elif method == "mode":
            fill_value = self.building_dataset[column_name].mode()[0]
        else:
            return "Invalid method. Please choose 'mean', 'median', or 'mode'."

        self.building_dataset[column_name].fillna(fill_value, inplace=True)

    def drop_missing_values(self):
        """
        Drops rows with any missing values in the dataset.
        """
        self.building_dataset.dropna(inplace=True)

    def convert_to_datetime(self, column_name, format_string="%d-%b-%Y"):
        """
        Converts a specified column to a datetime format.

        Parameters:
        ----------
        column_name : str
            The name of the column to convert.
        format_string : str, optional
            The format string to use for the conversion. Default is '%d-%b-%Y'.
        """
        try:
            self.building_dataset[column_name] = pd.to_datetime(
                self.building_dataset[column_name], format=format_string
            )
        except ValueError as e:
            return f"Conversion error: {e}"

    def remove_outliers(self, column_name, method="IQR"):
        """
        Removes outliers from a specified numeric column using the IQR or Z-score method.

        Parameters:
        ----------
        column_name : str
            The name of the numeric column to remove outliers from.
        method : str, optional
            The method to use for outlier detection ('IQR' or 'Z-score'). Default is 'IQR'.
        """
        if column_name not in self.building_dataset.columns:
            return f"Column '{column_name}' not found in the dataset."

        if method == "IQR":
            Q1 = self.building_dataset[column_name].quantile(0.25)
            Q3 = self.building_dataset[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.building_dataset = self.building_dataset[
                (self.building_dataset[column_name] >= lower_bound)
                & (self.building_dataset[column_name] <= upper_bound)
            ]
        elif method == "Z-score":
            from scipy import stats

            z_scores = stats.zscore(self.building_dataset[column_name])
            abs_z_scores = np.abs(z_scores)
            self.building_dataset = self.building_dataset[(abs_z_scores < 3)]
        else:
            return "Invalid method. Please choose 'IQR' or 'Z-score'."

    def clean_text_columns(self, column_name):
        """
        Cleans a specified text column by removing whitespace, standardizing case, and eliminating special characters.

        Parameters:
        ----------
        column_name : str
            The name of the text column to clean.
        """
        if column_name not in self.building_dataset.columns:
            return f"Column '{column_name}' not found in the dataset."

        self.building_dataset[column_name] = self.building_dataset[
            column_name
        ].str.strip()  # Remove leading/trailing whitespace
        self.building_dataset[column_name] = self.building_dataset[
            column_name
        ].str.lower()  # Convert to lowercase
        self.building_dataset[column_name] = self.building_dataset[
            column_name
        ].str.replace(
            "[^\w\s]", ""
        )  # Remove special chars
