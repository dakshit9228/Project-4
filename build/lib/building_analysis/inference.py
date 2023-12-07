import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Inference:
    """
    A class used to perform data aggregation, statistical analysis, and visualization on a dataset
    related to parking spaces.

    Attributes:
    ----------
    data : DataFrame
        A Pandas DataFrame containing the data to be analyzed.
    """

    def __init__(self, data):
        """
        Initializes the Inference class with data provided.

        Parameters:
        ----------
        data : DataFrame
            A Pandas DataFrame that contains the data for analysis and visualization.
        """

        self.data = data

    def aggregate_data(self):
        """
        Aggregates the data by 'Region Code' and computes the mean 'Total Parking Spaces' for each region.

        Returns:
        -------
        DataFrame
            A DataFrame with 'Region Code' as the index and the average 'Total Parking Spaces' for that region.
        """

        self.region_parking = (
            self.data.groupby("Region Code")["Total Parking Spaces"]
            .mean()
            .reset_index()
        )
        return self.region_parking

    def statistical_summary(self):
        """
        Provides a statistical summary of the aggregated region parking data.

        Returns:
        -------
        DataFrame
            A DataFrame with descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
            for the 'Total Parking Spaces' aggregated by region.
        """
        return self.region_parking.describe()

    def visualize_matplotlib(self):
        """
        Visualizes the average 'Total Parking Spaces' by 'Region Code' using a Matplotlib bar chart.

        The visualization is displayed as a 10x6 figure with 'Region Code' on the x-axis and
        'Average Total Parking Spaces' on the y-axis.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(
            self.region_parking["Region Code"],
            self.region_parking["Total Parking Spaces"],
        )
        plt.xlabel("Region Code")
        plt.ylabel("Average Total Parking Spaces")
        plt.title("Average Total Parking Spaces by Region (Matplotlib)")
        plt.show()

    def visualize_seaborn(self):
        """
        Visualizes the average 'Total Parking Spaces' by 'Region Code' using a Seaborn bar chart.

        The visualization is displayed as a 10x6 figure with 'Region Code' on the x-axis and
        'Average Total Parking Spaces' on the y-axis, leveraging the advanced styling capabilities of Seaborn.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Region Code", y="Total Parking Spaces", data=self.region_parking)
        plt.xlabel("Region Code")
        plt.ylabel("Average Total Parking Spaces")
        plt.title("Average Total Parking Spaces by Region (Seaborn)")
        plt.show()

    def process_and_visualize_heatmap(self):
        """
        Processes the 'Construction Date' to extract the year, groups the data by 'Region Code' and
        'Historical Status', and calculates the mean 'Total Parking Spaces', mean 'Bldg ANSI Usable',
        and median 'Construction Year'. It then visualizes this grouped data as a heatmap with 'Region Code'
        and 'Historical Status' as indices.

        The heatmap is displayed as a 12x8 figure with annotations, using a 'viridis' colormap to represent
        the magnitude of the values.
        """
        self.data["Construction Year"] = pd.to_datetime(
            self.data["Construction Date"]
        ).dt.year
        grouped_data = (
            self.data.groupby(["Region Code", "Historical Status"])
            .agg(
                {
                    "Total Parking Spaces": "mean",
                    "Bldg ANSI Usable": "mean",
                    "Construction Year": "median",
                }
            )
            .reset_index()
        )
        pivot_data = grouped_data.pivot_table(
            index=["Region Code", "Historical Status"],
            values=["Total Parking Spaces", "Bldg ANSI Usable", "Construction Year"],
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap="viridis")
        plt.title(
            "Heatmap of Average Parking Spaces and Usable Area by Region and Historical Status"
        )
        plt.ylabel("Region Code - Historical Status")
        plt.show()
