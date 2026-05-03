import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Đảm bảo đầu vào là mảng NumPy
    v=np.asanyarray(v)
    N=len(v)

    # Tạo ma trận kích thước N x N toàn số 0
    # dtype được lấy từ v để bảo toàn dữ liệu (float hoặc int)
    D = np.zeros((N, N), dtype=v.dtype)
    
    # Dùng vòng lặp điền các giá trị đường chéo chính
    for i in range(N):
        D[i, i]=v[i]
        
    return D
    
