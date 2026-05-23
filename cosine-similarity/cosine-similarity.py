import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # 1. Đảm bảo đầu vào là mảng NumPy 1D
    arr_a = np.asanyarray(a).flatten()
    arr_b = np.asanyarray(b).flatten()
    
    if arr_a.shape != arr_b.shape:
        raise ValueError("Hai vector phải có cùng kích thước và số chiều!")
        
    # 2. Tính tích vô hướng (Dot Product) của hai vector
    dot_product = np.dot(arr_a, arr_b)
    
    # 3. Tính độ lớn (L2 norm) của từng vector
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    
    # 4. Kiểm tra trường hợp vector không (Zero Vector) để tránh lỗi chia cho 0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    # 5. Áp dụng công thức: dot(a, b) / (norm(a) * norm(b))
    cosine_sim = dot_product / (norm_a * norm_b)
    
    # 6. Giới hạn giá trị trong khoảng [-1.0, 1.0] để tránh sai số dấu phẩy động (float precision)
    return float(np.clip(cosine_sim, -1.0, 1.0))