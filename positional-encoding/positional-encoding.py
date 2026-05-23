import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
   # 1. Khởi tạo ma trận kết quả với kích thước (seq_len, d_model)
    pe = np.zeros((seq_len, d_model))
    
    # 2. Xây dựng Column Vector chứa các vị trí: kích thước (T, 1)
    positions = np.arange(seq_len).reshape(-1, 1)
    
    # 3. Xây dựng Row Vector chứa các tần số: kích thước (1, ceil(d/2))
    # Số lượng cặp tần số cần thiết là ceil(d_model / 2)
    num_frequencies = int(np.ceil(d_model / 2))
    
    # Chỉ số 2i chạy từ 0, 2, 4,... với độ dài bằng num_frequencies
    i_indices = np.arange(0, num_frequencies * 2, 2).reshape(1, -1)
    
    # Tính toán thành phần tần số (frequencies) dựa trên biến đổi log để tránh tràn số
    frequencies = np.exp(i_indices * -(np.log(base) / d_model))
    
    # 4. Thực hiện phép nhân Broadcasting giữa (T, 1) và (1, ceil(d/2)) 
    # Kết quả của (positions * frequencies) sẽ là một ma trận kích thước (T, ceil(d/2))
    angle_rates = positions * frequencies
    
    # 5. Điền dữ liệu xen kẽ vào các cột của ma trận pe
    # Cột chẵn điền hàm SIN
    pe[:, 0::2] = np.sin(angle_rates[:, :pe[:, 0::2].shape[1]])
    
    # Cột lẻ điền hàm COS
    pe[:, 1::2] = np.cos(angle_rates[:, :pe[:, 1::2].shape[1]])
    
    return pe
   