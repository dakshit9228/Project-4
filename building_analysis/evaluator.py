from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    """
    ModelEvaluator - A library for evaluating regression models using common metrics.

    This library provides a class, ModelEvaluator, designed for evaluating regression models using standard metrics
    such as Mean Squared Error (MSE) and R-squared (R2). It includes a method for calculating these metrics based on
    the predictions and actual values.

    Methods:
        evaluate(self, model, X_test, y_test):
            Evaluates the regression model using the provided test data.

            Args:
                model: The trained regression model to be evaluated.
                X_test (array-like or pd.DataFrame): The feature values of the test set.
                y_test (array-like or pd.Series): The true target values of the test set.

            Returns:
                tuple: A tuple containing Mean Squared Error (MSE) and R-squared (R2) scores.

    Usage:
        # Instantiate the model evaluator
        evaluator = ModelEvaluator()

        # Evaluate the model
        mse, r2 = evaluator.evaluate(trained_model, test_features, test_targets)

    Author:
        Your Name

    Version:
        1.0.0
    """
    def evaluate(self, model, X_test, y_test):
        """
        Evaluates the regression model using the provided test data.

        Args:
            model: The trained regression model to be evaluated.
            X_test (array-like or pd.DataFrame): The feature values of the test set.
            y_test (array-like or pd.Series): The true target values of the test set.

        Returns:
            tuple: A tuple containing Mean Squared Error (MSE) and R-squared (R2) scores.
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, r2
