class BuildingDatasetEDA:
    """
    A simplified class for performing basic Exploratory Data Analysis (EDA) on the building dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def plot_histogram(self, column_name):
        """
        Plots a histogram for a specified numeric column.

        Parameters:
        column_name : str
            The name of the numeric column for which to generate the histogram.
        """
        if column_name in self.dataset.columns:
            self.dataset[column_name].hist()
            plt.title(f"Histogram of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_bar_chart(self, column_name):
        """
        Plots a bar chart for a specified categorical column.

        Parameters:
        column_name : str
            The name of the categorical column for which to generate the bar chart.
        """
        if column_name in self.dataset.columns:
            value_counts = self.dataset[column_name].value_counts()
            value_counts.plot(kind='bar')
            plt.title(f"Bar Chart of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Count")
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_boxplot(self, column_name):
        """
        Plots a boxplot for a specified numeric column.

        Parameters:
        column_name : str
            The name of the numeric column for which to generate the boxplot.
        """
        if column_name in self.dataset.columns:
            self.dataset.boxplot(column=column_name)
            plt.title(f"Boxplot of {column_name}")
            plt.ylabel(column_name)
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_scatterplot(self, column_x, column_y):
        """
        Plots a scatterplot for two specified numeric columns.

        Parameters:
        column_x, column_y : str
            The names of the numeric columns to use for the x and y axes of the scatterplot.
        """
        if column_x in self.dataset.columns and column_y in self.dataset.columns:
            self.dataset.plot.scatter(x=column_x, y=column_y)
            plt.title(f"Scatterplot of {column_x} vs {column_y}")
            plt.xlabel(column_x)
            plt.ylabel(column_y)
            plt.show()
        else:
            print(f"One or both columns '{column_x}', '{column_y}' not found in the dataset.")

    def plot_pie_chart(self, column_name):
        if column_name in self.dataset.columns:
            pie_data = self.dataset[column_name].value_counts()
            pie_data.plot(kind='pie', autopct='%1.1f%%')
            plt.title(f"Pie Chart of {column_name}")
            plt.ylabel('')  # Hide the y-label
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_line_graph(self, column_name):
        if column_name in self.dataset.columns:
            self.dataset[column_name].plot(kind='line')
            plt.title(f"Line Graph of {column_name}")
            plt.ylabel(column_name)
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_countplot(self, column_name):
        if column_name in self.dataset.columns:
            sns.countplot(x=column_name, data=self.dataset)
            plt.title(f"Count Plot of {column_name}")
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Column '{column_name}' not found in the dataset.")

    def plot_correlation_heatmap(self):
        correlation_matrix = self.dataset.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

