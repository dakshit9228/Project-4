import pandas as pd

class BuildingDatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.building_dataset = None

    def load_building_dataset(self):
        try:
            self.building_dataset = pd.read_csv(self.file_path)
            return self.building_dataset
        except FileNotFoundError:
            return f"Error: File not found at {self.file_path}"

    def get_building_dataset(self):
        if self.building_dataset is not None:
            return self.building_dataset
        else:
            return "Error: Building dataset not loaded. Use load_building_dataset() method first."
