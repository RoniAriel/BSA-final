import os

import pandas as pd


class DataProcessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data_dict = {}
        self.df = None

    def load_data(self):
        """
        Load the CSV file into a DataFrame if the file exists.
        """
        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError(f"File '{self.csv_file}' not found.")

        self.df = pd.read_csv(self.csv_file)

    def process_data(self):
        """
        Process the data and store it in a dictionary.
        """
        if self.df is None:
            raise ValueError("DataFrame not loaded. Call load_data first.")

        for col in self.df.columns:
            indices = [idx for idx, val in enumerate(self.df[col]) if pd.notna(val)]
            self.data_dict[col] = indices

    def get_data_dict(self):
        """
        Get the processed data dictionary.
        """
        if not self.data_dict:
            raise ValueError("Data dictionary is empty. Call process_data first.")

        return self.data_dict



