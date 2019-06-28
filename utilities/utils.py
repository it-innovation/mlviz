import numpy as np

def is_binary(x):
    """
    Checks if input array contains only binary values.
        
    Parameters:
        x (np.ndarry): array containing ints or floats.
        
    Returns:
        bool
    """
    try:
        is_x_binary = np.array_equal(x, x.astype(bool))
        return is_x_binary
    except AttributeError:
        # x is not a numpy array
        return False


def is_categorical(df, threshold=0.1, remove=False):
    """
    Returns the names of columns which could be categorical, 
    as determined by the fraction of unique values of the 
    feature for all training instances.

    Parameters:
        df (pd.DataFrame): training set.

    Returns:
        idxs (pd.Series): names of potentially categorical features.
    """
    unique_values = df.nunique() / df.shape[0]
    
    idxs = np.argsort(unique_values)
    
    return idxs[unique_values[idxs]<threshold].index