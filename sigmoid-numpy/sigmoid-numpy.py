import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x=np.asarray(x, dtype=float)

    # Formula 1/(1+exp(-x))
    return 1/(1+np.exp(-x))
    