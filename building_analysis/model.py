from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

class BuildingRegressionModel:
    """
    BuildingRegressionModel - A library for building regression models on building-related datasets.

    This library provides a class, BuildingRegressionModel, designed for creating regression models on datasets
    related to building information. It includes methods for creating a pipeline, incorporating preprocessing steps
    and linear regression modeling.

    Attributes:
        dataset (pd.DataFrame): The input dataset for building the regression model.
        pipeline (Pipeline): The scikit-learn pipeline that includes preprocessing and regression steps.

    Methods:
        __init__(self, dataset):
            Initializes the BuildingRegressionModel with the provided dataset.

        create_pipeline(self, numerical_cols, categorical_cols):
            Creates a scikit-learn pipeline for building a regression model. The pipeline includes:
                - Standard scaling for numerical columns.
                - One-hot encoding for categorical columns.
                - Linear regression as the regression model.

            Args:
                numerical_cols (list): List of column names containing numerical features.
                categorical_cols (list): List of column names containing categorical features.

            Returns:
                Pipeline: The scikit-learn pipeline.

            Raises:
                ValueError: If specified columns are missing in the dataset.

    Usage:
        # Instantiate the regression model with a dataset
        regression_model = BuildingRegressionModel(my_dataset)

        # Create the pipeline
        pipeline = regression_model.create_pipeline(numerical_cols=['Num1', 'Num2'],
                                                    categorical_cols=['Cat1', 'Cat2'])

    Author:
        Your Name

    Version:
        1.0.0
    """
    def __init__(self, dataset):
        """
        Initializes the BuildingRegressionModel with the provided dataset.

        Args:
            dataset (pd.DataFrame): The input dataset for building the regression model.
        """
        self.dataset = dataset
        self.pipeline = None

    def create_pipeline(self, numerical_cols, categorical_cols):
        """
        Creates a scikit-learn pipeline for building a regression model.

        The pipeline includes:
            - Standard scaling for numerical columns.
            - One-hot encoding for categorical columns.
            - Linear regression as the regression model.

        Args:
            numerical_cols (list): List of column names containing numerical features.
            categorical_cols (list): List of column names containing categorical features.

        Returns:
            Pipeline: The scikit-learn pipeline.

        Raises:
            ValueError: If specified columns are missing in the dataset.
        """
        # Check for missing columns
        missing_cols = [col for col in numerical_cols + categorical_cols if col not in self.dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in the dataset: {missing_cols}")

        # Create the preprocessing transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        # Create the pipeline
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())])
        return self.pipeline
