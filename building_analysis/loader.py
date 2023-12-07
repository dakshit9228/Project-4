import pandas as pd


class BuildingDatasetLoader:
    """
    A class responsible for loading a building dataset from a specified file path.

    Attributes:
    ----------
    file_path : str
        The file path to the dataset file.
    building_dataset : DataFrame or None
        The variable that will hold the loaded building dataset after calling load_building_dataset method.
        Initialized to None until the dataset is loaded.
    """

    def __init__(self, file_path):
        """
        Initializes the BuildingDatasetLoader with the provided file path.

        Parameters:
        ----------
        file_path : str
            The path to the CSV file that contains the building dataset.
        """

        self.file_path = file_path
        self.building_dataset = None

    def load_building_dataset(self):
        """
        Loads the building dataset from the CSV file specified by the file_path attribute.

        Returns:
        -------
        DataFrame or str
            The building dataset as a Pandas DataFrame if successful, or an error string indicating
            that the file was not found.
        """

        try:
            self.building_dataset = pd.read_csv(self.file_path)
            return self.building_dataset
        except FileNotFoundError:
            return f"Error: File not found at {self.file_path}"

    def get_building_dataset(self):
        """
        Retrieves the building dataset if it has been loaded.

        Returns:
        -------
        DataFrame or str
            The building dataset as a Pandas DataFrame if it is already loaded, otherwise an error
            string stating that the dataset is not loaded and suggesting to call the
            load_building_dataset() method.
        """

        if self.building_dataset is not None:
            return self.building_dataset
        else:
            return "Error: Building dataset not loaded. Use load_building_dataset() method first."
