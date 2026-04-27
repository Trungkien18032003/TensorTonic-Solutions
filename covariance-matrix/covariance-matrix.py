import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X=np.asarray(X)

    if X.ndim != 2 or  X.shape[0] < 2:
        return None
        
    N=X.shape[0]

    mean_u = np.mean(X,axis=0)
    X_centered = X -mean_u

    covariance_matrix = (1/(N-1) * (X_centered.T @ X_centered))
    return covariance_matrix

X=[[1, 2], [2, 3], [3, 4]]
X1=[[1, 0], [0, 1]]
X2=[[1, 2, 3]]
print("Kết quả tối ưu:\n", covariance_matrix(X))

print("Kết quả tối ưu:\n", covariance_matrix(X1))

print("Kết quả tối ưu:\n", covariance_matrix(X2))