import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # Write code here
    N = len(documents)
    if N == 0:
        return {"tfidf_matrix": np.empty((0, 0)), "vocabulary": []}
    
    # 1. Tokenize: chuyển chữ thường và tách từ theo khoảng trắng
    doc_tokens = []
    df_counter = Counter()
    
    for doc in documents:
        tokens = doc.lower().split()
        doc_tokens.append(tokens)
        # Hint 1: Dùng set(tokens) để đếm số văn bản chứa từ (df)
        df_counter.update(set(tokens))
    
    # 2. Xây dựng và sắp xếp từ vựng (vocabulary) theo thứ tự bảng chữ cái
    vocabulary = sorted(list(df_counter.keys()))
    V = len(vocabulary)
    
    if V == 0:
        return {"tfidf_matrix": np.zeros((N, 0)), "vocabulary": []}
    
    # Hint 2: Tạo map từ -> chỉ số cột
    token_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    
    # Hint 3: Khởi tạo ma trận kết quả shape (N, V) bằng np.zeros
    tfidf_matrix = np.zeros((N, V), dtype=float)
    
    # Tính trước IDF cho từng từ: idf(t) = log(N / df(t)) (unsmoothed natural-log)
    idf = {term: math.log(N / df_counter[term]) for term in vocabulary}
    
    # 3. Tính toán TF-IDF cho từng tài liệu
    for doc_idx, tokens in enumerate(doc_tokens):
        total_tokens = len(tokens)
        if total_tokens == 0:
            continue
        
        # Đếm tần suất xuất hiện của từ trong tài liệu d
        tf_counts = Counter(tokens)
        
        for term, count in tf_counts.items():
            col_idx = token_to_idx[term]
            # tf(t, d) = count(t, d) / |d|
            tf = count / total_tokens
            # tfidf(t, d) = tf(t, d) * idf(t)
            tfidf_matrix[doc_idx, col_idx] = tf * idf[term]
            
    return {
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary
    }
  