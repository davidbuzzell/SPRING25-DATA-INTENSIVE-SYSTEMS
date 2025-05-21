from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(X: np.array, test_size: float=0.2, 
                random_state: float = 42, shuffle: bool = True):
    """Split the data into training and testing sets.

    NOTE:
        Please use the train_test_split function from sklearn to split the data.
        Ensure your test_size is set to 0.2.
        Ensure your random_state is set to 42.
        Ensure shuffle is set to True.

    Args:
        X (np.array): The independent variables.
        test_size (float): The proportion of the data to use for testing.
        random_state (int): The random seed to use for the split.
        shuffle (bool): Whether or not to shuffle the data before splitting.

    Returns:
        Tuple[np.array, np.array]: The training and testing sets.
            - x_train: The training independent variables.
            - x_test: The testing independent variables.

    """

    # 1. Use the train_test_split function from sklearn to split the data.
    return train_test_split(X,
                            test_size=test_size,
                            random_state=random_state,
                            shuffle=shuffle)

def standardize(x_train: np.array, x_test: np.array) -> Tuple[np.array, np.array]:
    """Standardize the dataset using StandardScaler from sklearn.

    NOTE: You should use the StandardScaler from sklearn to standardize the dataset. StandardScaler
    will standardize the dataset to have a mean of 0 and a standard deviation of 1, and operates
    on each feature independently.

    Args:
        x_train (np.array): The training dataset.
        x_test (np.array): The testing dataset.

    Returns:
        Tuple[np.array, np.array]: The standardized training and testing datasets.
    """

    # 1. Create a StandardScaler object.
    scaler = StandardScaler()

    # 2. Fit the scaler to the training data using scaler.fit.
    fit_scaler = scaler.fit(x_train)


    # 3. Transform the training data using scaler.transform.
    train_data = fit_scaler.transform(x_train)

    # 4. Transform the testing data using scaler.transform.
    test_data = fit_scaler.transform(x_test)


    # 5. Return the standardized datasets and the scaler.
    return train_data, test_data


def linear_regression(X: np.array, y: np.array) -> np.array:
    """Perform linear regression using the normal equation.

    NOTE: It is important that you concatenate a column of ones to the independent
    variables X. This will effectively add a bias term to the model.

    Args:
        X (np.array): The independent variables.
        y (np.array): The dependent variables.
    
    Returns:
        np.array: The weights for the linear regression model (including the bias term)
    """

    # 1. Concatenate the bias term to X using np.hstack.
    # NOTE: By convention, the bias term is the first column of the weights.
    nrows, _ = X.shape
    X_bias = np.hstack((X, np.ones((nrows, 1))))

    # 2. Calculate the weights using the normal equation.
    theta = np.matmul(np.linalg.inv(np.matmul(X_bias.transpose(), X_bias)),
                      np.matmul(X_bias.transpose(), y))

    # 3. Return the weights.
    return theta


def linear_regression_predict(X: np.array, weights: np.array) -> np.array:
    """Predict the dependent variables using the weights and independent variables.

    NOTE: It is important that you concatenate a column of ones to the independent
    variables X. This will effectively add a bias term to the model.

    Args:
        X (np.array): The independent variables.
        weights (np.array): The weights of the linear regression model.
    
    Returns:
        np.array: The predicted dependent variables.
    """
    # 1. Concatenate the bias term to X using np.hstack.
    # By convention, the bias term is the first column of the weights.
    nrows, _ = X.shape
    X_bias = np.hstack((X, np.ones((nrows, 1))))
    
    # 2. Calculate the predictions.
    y_hat = np.matmul(X_bias, weights)

    # 3. Return the predictions.
    return y_hat
    
def mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Calculate the mean squared error.

    You should use only numpy for this calculation.

    Args:
        y_true (np.array): The true values.
        y_pred (np.array): The predicted values.
    
    Returns:
        float: The mean squared error.
    """
    return np.sum((y_true - y_pred) ** 2) / y_true.shape[0]
