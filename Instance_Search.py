import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

class BOVWSearcher:
    def __init__(self, database_path):
        print("Đang load BOVW database...")
        with open(database_path, 'rb') as f:
            data = pickle.load(f)
            self.database = data['database']
            self.vocabulary = data['vocabulary']
            self.n_clusters = data['n_clusters']
            self.idf_weights = data.get('idf_weights', np.ones(self.n_clusters))
        
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            n_jobs=1,
            verbose=0
        )
        self.kmeans.cluster_centers_ = self.vocabulary
        self.kmeans.fit(self.vocabulary)
        
        print(f"Đã load database với {len(self.database)} ảnh")

    def process_query_image(self, image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is None:
                return None
                
            # Tạo BOVW histogram tương tự như trong training
            descriptors = normalize(descriptors, norm='l2', axis=1)
            distances = self.kmeans.transform(descriptors)
            
            sigma = np.mean(distances) / 2
            weights = np.exp(-distances / (2 * sigma**2))
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            histogram = np.zeros(self.n_clusters)
            
            if keypoints:
                kp_weights = np.array([kp.size * kp.response for kp in keypoints])
                kp_weights = kp_weights / np.sum(kp_weights)
                for i in range(len(descriptors)):
                    histogram += weights[i] * kp_weights[i]
            else:
                histogram = weights.sum(axis=0)
            
            histogram *= self.idf_weights
            histogram = normalize(histogram.reshape(1, -1), norm='l2')[0]
            
            return histogram
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            return None

    def search_image(self, query_image, top_k=5):
        query_features = self.process_query_image(query_image)
        if query_features is None:
            return []
            
        results = []
        for image_name, features in self.database.items():
            similarity = 1 - cosine(query_features, features['histogram'])
            results.append({
                'image_name': image_name,
                'score': similarity,
                'image': features['image']
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

def main():
    st.set_page_config(page_title="Image Search Demo", layout="wide")
    
    # Phần 1: Giới thiệu Dataset
    st.title("Demo Hệ thống Tìm kiếm Ảnh")
    st.header("1. Giới thiệu Dataset")
    st.write("""
    Dataset gồm 5000 ảnh đa dạng được thu thập từ COCO dataset, bao gồm:
    - Các đối tượng thường gặp trong cuộc sống hàng ngày
    - Phong cảnh thiên nhien
    - Con người và động vật
    - Đồ vật, phương tiện giao thông
    - Các hoạt động và sự kiện
    
    Mỗi ảnh có kích thước và nội dung khác nhau, giúp đánh giá hiệu quả của hệ thống tìm kiếm trong nhiều tình huống khác nhau.
    """)
    
    # Danh sách tên các ảnh mẫu
    sample_images = [
        "DTS/000000001675.jpg", "DTS/000000001761.jpg", "DTS/000000001818.jpg", "DTS/000000001993.jpg", "DTS/000000002006.jpg",
        "DTS/000000002149.jpg", "DTS/000000002153.jpg", "DTS/000000002157.jpg", "DTS/000000002261.jpg", "DTS/000000002299.jpg"
    ]
    
    # Tạo 2 hàng, mỗi hàng 5 cột để hiển thị ảnh mẫu
    for row in range(2):
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            image_index = row * 5 + idx
            with col:
                try:
                    st.image(sample_images[image_index], 
                            caption=f"Ảnh mẫu {image_index + 1}",
                            use_column_width=True)
                except Exception as e:
                    st.error(f"Không thể load ảnh {sample_images[image_index]}")
    
    # Phần 2: Giới thiệu Quy trình
    st.header("2. Quy trình xử lý BOVW")
    
    # SIFT Feature Extraction
    st.subheader("2.1. Trích xuất đặc trưng SIFT")
    st.write("""
    - Sử dụng SIFT để trích xuất keypoints và descriptors từ ảnh
    - Mỗi keypoint chứa thông tin về vị trí, scale, và orientation
    - Mỗi descriptor là vector 128 chiều
    """)
    st.image("SIFT-feature-extraction-algorithm-process.png", 
             caption="SIFT keypoints và descriptors", 
             use_column_width=True)
    
    # Visual Vocabulary Construction
    st.subheader("2.2. Xây dựng Vocabulary")
    st.write("""
    - Thu thập tất cả SIFT descriptors từ dataset
    - Sử dụng K-means clustering để tạo visual words
    - Số lượng clusters = 1000 (có thể điều chỉnh)
    """)
    st.image("The-features-extraction-system-using-bag-of-visual-words-BoVW.png", 
             caption="K-means clustering visual words", 
             use_column_width=True)
    
    st.subheader("2.3. Các kỹ thuật trong BOVW Histogram")  
    st.markdown("""
    ### Quy trình chi tiết thuật toán Bag of Visual Words (BOVW):

    #### 1. Trích xuất đặc trưng
    - Phát hiện các điểm đặc trưng (keypoints) trên ảnh sử dụng SIFT/SURF
    - Mỗi keypoint được mô tả bằng vector đặc trưng 128 chiều (SIFT descriptor)
    - Descriptor mô tả gradient theo 8 hướng trong 16 vùng 4x4 xung quanh keypoint
    - Chuẩn hóa descriptor để bất biến với độ sáng và độ tương phản

    #### 2. Xây dựng từ điển thị giác (Visual Vocabulary)
    - Thu thập tất cả descriptor từ tập ảnh huấn luyện (thường là hàng triệu descriptor)
    - Sử dụng K-means clustering để phân cụm các descriptor thành K cụm (ví dụ K=1000)
    - Mỗi tâm cụm trở thành một "visual word" trong từ điển
    - Lưu từ điển để tái sử dụng cho các ảnh mới

    #### 3. Mã hóa đặc trưng (Feature Encoding)
    - Hard Assignment: Gán mỗi descriptor cho visual word gần nhất
    - Soft Assignment: Gán descriptor cho nhiều visual word với trọng số khác nhau
    - Tạo histogram chuẩn hóa L1 hoặc L2 thể hiện tần suất của visual words

    #### 4. Áp dụng trọng số TF-IDF
    - Term Frequency (TF): Tần suất xuất hiện của visual word trong ảnh
    - Inverse Document Frequency (IDF): log(N/n_i)
        + N: Tổng số ảnh trong dataset
        + n_i: Số ảnh chứa visual word thứ i
    - Trọng số cuối = TF × IDF
    - IDF giúp giảm tầm quan trọng của visual words phổ biến, tăng tầm quan trọng của visual words đặc trưng

    #### 5. Tối ưu hóa biểu diễn
    - Chuẩn hóa L2 cho vector đặc trưng cuối cùng
    - Áp dụng PCA để giảm chiều dữ liệu
    - Sử dụng Spatial Pyramid Matching để bảo toàn thông tin không gian
    - Lượng tử hóa vector để giảm dung lượng lưu trữ

    #### 6. Tìm kiếm và so khớp
    - Với ảnh truy vấn, thực hiện các bước 1-5 để có vector đặc trưng
    - Tính độ tương đồng bằng cosine similarity hoặc khoảng cách Euclidean
    - Sử dụng cấu trúc dữ liệu hiệu quả (như k-d tree) để tăng tốc tìm kiếm
    - Xếp hạng và trả về top-K ảnh tương đồng nhất

    > BOVW kết hợp với TF-IDF và các kỹ thuật tối ưu giúp biểu diễn ảnh hiệu quả,
    > cho phép tìm kiếm chính xác và nhanh chóng trong tập dữ liệu lớn.
    """)
    
    # Thêm một ví dụ tương tác
    
    # Phần 3: Instance Search
    st.header("3. Instance Search")
    
    # Thêm sidebar cho các tùy chọn
    st.sidebar.title("Tùy chọn tìm kiếm")
    top_k = st.sidebar.slider("Số lượng kết quả", min_value=1, max_value=20, value=5)
    
    # Thêm thông tin về ứng dụng trong sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Về ứng dụng")
    st.sidebar.write("""
    Ứng dụng sử dụng:
    - SIFT để trích xuất đặc trưng
    - BOVW để biểu diễn ảnh
    - Cosine similarity để so sánh
    """)
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh để tìm kiếm...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Hiển thị ảnh query và kết quả trong 2 cột
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Ảnh Query")
            query_image = Image.open(uploaded_file)
            st.image(query_image, use_column_width=True)
        
        # Xử lý ảnh và tìm kiếm
        try:
            query_array = np.array(query_image)
            if len(query_array.shape) == 2:
                query_array = cv2.cvtColor(query_array, cv2.COLOR_GRAY2BGR)
            elif query_array.shape[2] == 4:
                query_array = cv2.cvtColor(query_array, cv2.COLOR_RGBA2BGR)
            
            with st.spinner('Đang tìm kiếm...'):
                searcher = BOVWSearcher("bovw_database_compressed.pkl")
                results = searcher.search_image(query_array, top_k=top_k)
            
            # Hiển thị kết quả
            with col2:
                st.subheader("Kết quả tìm kiếm")
                if results:
                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            st.image(result['image'],
                                   caption=f"Score: {result['score']:.3f}\n{result['image_name']}",
                                   use_column_width=True)
                else:
                    st.warning("Không tìm thấy ảnh tương tự!")
                    
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()