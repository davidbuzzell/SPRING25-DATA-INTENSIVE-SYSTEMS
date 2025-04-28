import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def impute_numerical_values(X: np.ndarray) -> np.ndarray:
    """Impute missing values in a numpy array using IterativeImputer with RandomForestRegressor.

    NOTE: Call the sklearn functions with the following parameters:
        - estimator: RandomForestRegressor with n_estimators=10 and random_state=0
        - imputer: IterativeImputer with max_iter=10.

    Args:
        X (numpy.ndarray): Input data with missing values.
    
    Returns:
        numpy.ndarray: Imputed data with missing values filled in.
    """

    # 1. Init a RandomForestRegressor for the imputer. 
    # Make sure you use 10 estimators and set the random state to 0.
 

    # 2. Init the imputer with the estimator. Use max_iter=10.


    # 3. Fit the imputer on the dataset.


    # 4. Transform the dataset to impute missing values.


    raise NotImplementedError("Please implement the impute_numerical_values function in preprocessing.py.")



def standard_scale_with_numpy(x: np.ndarray) -> np.ndarray:
    """Scale data using numpy.

    Without using sklearn, we can scale the data using numpy by 
    subtracting the mean of the array and dividing by the standard deviation.

    NOTE: Numpy has built-in functions to calculate the mean and standard deviation,
    which can be accessed through x.mean() and x.std() respectively.

    Args:
        x (numpy.ndarray): Input data to be scaled.
    
    Returns:
        numpy.ndarray: Scaled data.
    """
    raise NotImplementedError("Please implement the standard_scale_with_numpy function in preprocessing.py.")



def minmax_scale_with_numpy(x: np.ndarray) -> np.ndarray:
    """Scale data using numpy.

    Without using sklearn, we can scale the data using numpy by 
    subtracting the min and dividing by the max minus min.

    NOTE: Numpy has built-in functions to calculate the min and max,
    which can be accessed through x.min() and x.max() respectively.

    Args:
        x (numpy.ndarray): Input data to be scaled.
    
    Returns:
        numpy.ndarray: Scaled data.
    """
    raise NotImplementedError("Please implement the minmax_scale_with_numpy function in preprocessing.py.")



def binarize_islands(islands: list[str]) -> list[int]:
    """Convert a list of island names to binary values.

    NOTE: Please use 1 for 'Biscoe', 0 for 'Adelie'. 
    Return a list of ints, not a numpy array.

    Args:
        islands (list[str]): List of islands names.
    
    Returns:
        list[int]: List of binary values.
    """
    raise NotImplementedError("Please implement the binarize_islands function in preprocessing.py.")



def generate_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one-hot encoding for the 'species' column in the DataFrame.

    NOTE: Use the `pd.get_dummies` function to create one-hot encoded columns 
    for the 'species' column. Please ensure you use the `dtype` parameter to set the data 
    type of the new columns to 'int'. The end result should be a DataFrame like this:

    | species_Adelie | species_Chinstrap | species_Gentoo |
    |----------------|-------------------|----------------|
    | 0              | 1                 | 0              |

    
    Args:
        df (pd.DataFrame): The input DataFrame containing the 'species' column.
        
    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns for each species.
    """
    raise NotImplementedError("Please implement the generate_one_hot_encoding function in preprocessing.py.")



def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder the columns of the DataFrame to match convention.

    By convention, we want to have our label columns at the
    end of the DataFrame. Let's reorder the columns
    before we save the DataFrame to a CSV.

    Args:
        df (pd.DataFrame): The input DataFrame to reorder.
    
    Returns:
        pd.DataFrame: The reordered DataFrame with label columns at the end.
    """

    # Do not change the order of these columns.
    NEW_COLUMN_ORDER = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species_Adelie",
        "species_Chinstrap",
        "species_Gentoo",
        "island"
    ]

    # Reorder the columns of the DataFrame to match the new column order.
    
    raise NotImplementedError("Please implement the reorder_columns function in preprocessing.py.")