import numpy as np

def matrix_inverse(A):
    """
    Tính ma trận nghịch đảo A^-1 sao cho AA^-1 = I.
    Trả về None nếu ma trận không vuông hoặc suy biến (singular).
    """
    # Chuyển đổi đầu vào thành NumPy array và sử dụng kiểu số thực (float)
    # np.asanyarray giúp xử lý linh hoạt cả list và mảng NumPy
    A = np.asanyarray(A, dtype=float)
    
    # 1. Kiểm tra điều kiện ma trận vuông (n x n)
    # Ma trận phải có số hàng và số cột bằng nhau
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    
    n = A.shape[0]
    
    # 2. Tạo ma trận đơn vị I (Identity matrix)
    # Ma trận đơn vị có các số 1 trên đường chéo và 0 ở các vị trí khác
    I = np.eye(n)
    
    # 3. Tạo ma trận bổ sung [A | I]
    # Ghép ma trận A và I theo chiều ngang để thực hiện biến đổi đồng thời
    aug = np.hstack((A, I))
    
    # 4. Thực hiện thuật toán Gauss-Jordan
    for i in range(n):
        # Tìm phần tử trụ (pivoting) để tránh chia cho 0 và tăng độ chính xác
        pivot_row = np.argmax(np.abs(aug[i:, i])) + i
        
        # Kiểm tra tính suy biến (Singular): det(A) = 0
        # Nếu phần tử trụ lớn nhất xấp xỉ bằng 0, ma trận không có nghịch đảo
        if np.abs(aug[pivot_row, i]) < 1e-12:
            return None
        
        # Hoán đổi hàng hiện tại với hàng chứa phần tử trụ
        aug[[i, pivot_row]] = aug[[pivot_row, i]]
        
        # Chuẩn hóa hàng i: Chia hàng i cho phần tử trụ để đưa giá trị tại đường chéo về 1
        aug[i] = aug[i] / aug[i, i]
        
        # Khử các phần tử khác tại cột i trên các hàng j khác i để đưa chúng về 0
        for j in range(n):
            if i != j:
                # Sử dụng phép toán mảng (array operations) để tối ưu tốc độ
                aug[j] -= aug[j, i] * aug[i]
                
    # 5. Trích xuất ma trận kết quả
    # Ma trận nghịch đảo A^-1 chính là phần bên phải của ma trận bổ sung sau khi biến đổi
    return aug[:, n:]

# --- Kiểm tra các trường hợp ---

def run_test():
    # Ví dụ 1: Ma trận khả nghịch 2x2
    A1 = [[4, 7], [2, 6]]
    print("Ma trận A1:\n", np.array(A1))
    print("Nghịch đảo A1:\n", matrix_inverse(A1))
    print("-" * 20)

    # Ví dụ 2: Ma trận suy biến (Singular) - det = 0
    # Các hàng phụ thuộc tuyến tính sẽ dẫn đến định thức bằng 0
    A2 = [[1, 2], [2, 4]]
    print("Ma trận A2 (Suy biến):")
    print(matrix_inverse(A2)) # Kỳ vọng: None
    print("-" * 20)

    # Ví dụ 3: Ma trận không vuông
    A3 = [[1, 2, 3], [4, 5, 6]]
    print("Ma trận A3 (Không vuông):")
    print(matrix_inverse(A3)) # Kỳ vọng: None

if __name__ == "__main__":
    run_test()