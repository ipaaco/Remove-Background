import warnings
import numpy as np

def softmax(x, axis=None):
    """
    Computes the softmax of an input array.

    Parameters
    ----------
    x : numpy array
        The input array for which to compute the softmax.
    axis : int or None, optional
        The axis along which to apply the softmax function. If None, softmax 
        is applied to the entire array.

    Returns
    -------
    f_x : numpy array
        The resulting array after applying the softmax function.
    """
    # Find the maximum value along the specified axis for numerical stability
    max = np.max(x, axis=axis, keepdims=True)
    
    # Compute the exponentials, shifting by the max value to prevent overflow
    e_x = np.exp(x - max)
    
    # Sum of exponentials along the specified axis
    sum = np.sum(e_x, axis=axis, keepdims=True)
    
    # Divide each exponential by the sum to get the softmax probabilities
    f_x = e_x / sum
    return f_x

def sigmoid(x):
    """
    Computes the sigmoid of an input array.

    Parameters
    ----------
    x : numpy array
        The input array for which to compute the sigmoid function.

    Returns
    -------
    numpy array
        The resulting array after applying the sigmoid function.
    """
    # Suppress warnings for numerical issues in the exponential calculation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # Compute the sigmoid function element-wise
        return 1.0 / (1.0 + np.exp(-x))
