import numpy as np

def matrix_transpose(A):
    # Đảm bảo đầu vào là NumPy array để lấy được .shape
    A = np.asanyarray(A)
    
    # 1. Lấy kích thước (N hàng, M cột)
    N, M = A.shape
    
    # 2. Khởi tạo ma trận mới kích thước (M, N)
    # Sử dụng np.empty sẽ nhanh hơn np.zeros một chút vì không cần xóa bộ nhớ
    #A_T = np.zeros((M, N), dtype=A.dtype) 
    
    A_T = np.empty((M, N), dtype=A.dtype) # Hàm np.empty được sử dụng để tạo một mảng NumPy mới có hình dạng và                                                kiểu dữ liệu cụ thể mà không cần khởi tạo các phần tử của nó.
     
   
    # Sử dụng vòng lặp thủ công để hoán đổi theo logic (i, j) -> (j, i)
    
    #for i in range(N):
    #    for j in range(M):
    #        A_T[j, i] = A[i, j]
    
    # Thay vì duyệt từng phần tử, ta lấy nguyên hàng i của A gán vào cột i của A_T
    for i in range(N):
        A_T[:, i] = A[i, :] # Gán hàng i thành cột i
            
    return A_T

# --- Kiểm tra hiệu năng ---
A_example = [[1, 2, 3], [4, 5, 6]]
print("Kết quả tối ưu:\n", matrix_transpose(A_example))