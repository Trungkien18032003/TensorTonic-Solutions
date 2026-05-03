import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Đảm bảo A: 2D NumPy array, shape(N,N)- square matrix (ma trận vuông)
    A=np.asanyarray(A)

    # Kiểm tra ma trận vuông 2 chiều N = N không
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Trace is defined only for square matrices!")
    # Lấy kích thước ma trận (N=N)
    N=A.shape[0]

    # Khởi tạo biến lưu tổng
    trace_sum=0

    # Duyệt qua các chỉ số và cộng dồn các phân tử đường chéo A[i,i]
    for i in range(N):
        trace_sum += A[i,i]  # trace_sum = trace_sum + A[i, i]
        
    return trace_sum
    
