import streamlit as st
import os
from glob import glob
import base64
import random

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_random_image_from_shape_type(angle_dir, shape_type_dir):
    """Lấy một hình ảnh ngẫu nhiên từ thư mục shape_type"""
    # Tạo đường dẫn đầy đủ đến thư mục shape_type
    full_path = os.path.join(angle_dir, shape_type_dir)
    # Lấy tất cả các file trong thư mục
    image_files = []
    if os.path.exists(full_path):
        # Lấy tất cả các file trong thư mục
        image_files = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Tạo đường dẫn đầy đủ cho mỗi file
        image_files = [os.path.join(full_path, f) for f in image_files]
    
    return random.choice(image_files) if image_files else None

def main():
    st.set_page_config(page_title="SuperPoint Analysis", layout="wide")
    
    # Tiêu đề chính
    st.title("Phân tích và đánh giá SuperPoint")
    
    # 1. Phần giới thiệu dataset
    st.header("1. Giới thiệu Dataset")
    
    # Container cho 8 hình ảnh phía trên
    st.write("#### Hình ảnh minh họa các keypoints được phát hiện")
    
    # Tạo 7 cột cho mỗi hàng
    cols_top = st.columns(7)
    
    # Danh sách đường dẫn đến các hình ảnh của bạn
    image_paths = [
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_checkerboard_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_cube_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_polygon_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_lines_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_multiple_polygons_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_star_original.png",
        "UIUX/FeaturesMatching/detector_results5/original_images/draw_stripes_original.png",
        # Hàng 2
        "UIUX/FeaturesMatching/detector_results5/draw_checkerboard_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_cube_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_polygon_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_lines_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_multiple_polygons_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_star_result.png",
        "UIUX/FeaturesMatching/detector_results5/draw_stripes_result.png",
    ]
    
    # Hiển thị hàng đầu tiên (7 ảnh)
    for i, col in enumerate(cols_top):
        with col:
            st.image(
                image_paths[i],
                caption=f"Ảnh gốc {i + 1}",
                use_column_width=True
            )
    
    # Tạo 7 cột cho hàng thứ hai
    cols_bottom = st.columns(7)
    
    # Hiển thị hàng thứ hai (7 ảnh)
    for i, col in enumerate(cols_bottom):
        with col:
            st.image(
                image_paths[i + 7],  # Lấy 7 ảnh tiếp theo
                caption=f"Keypoints Groundtruth {i + 8}",
                use_column_width=True
            )
    
    # Nội dung giới thiệu dataset
    st.write("""
    ### Tổng quan về Synthetic Dataset
    
    Dataset tổng hợp được tạo ra để huấn luyện và đánh giá mô hình SuperPoint, bao gồm nhiều loại hình học cơ bản khác nhau:

    1. **Cấu trúc Dataset:**
    Dataset bao gồm các thư mục con sau: draw_checkerboard, draw_cube, draw_ellipses, draw_lines, draw_multiple_polygons, draw_polygon, draw_star, draw_stripes, gaussian_noise

    Mỗi thư mục con đều chứa:
    - Thư mục `images/`: Lưu trữ các ảnh synthetic
    - Thư mục `points/`: Chứa các file .npy tương ứng lưu tọa độ groundtruth keypoints của mỗi ảnh
    
    2. **Đặc điểm của Dataset:**
    - Mỗi thư mục con đại diện cho một loại hình học khác nhau (đường thẳng, đa giác, hình sao,...)
    - Tất cả ảnh được tạo tự động (synthetic) với các keypoints được xác định chính xác
    - Mỗi ảnh đều có một file .npy tương ứng chứa thông tin về vị trí các keypoints

    3. **Format dữ liệu:**
    - Ảnh được lưu dưới dạng grayscale với kích thước 240x320 pixels
    - File .npy chứa mảng numpy 2 chiều với shape (N, 2), trong đó N là số lượng keypoints
    - Mỗi keypoint được biểu diễn bởi tọa độ (x, y) trong ảnh
    """)
    

    # 2. Phần phương pháp
    st.header("2. Kiến trúc SuperPoint")

    st.write("""
    SuperPoint là một mô hình deep learning được giới thiệu vào năm 2018 trong bài báo "SuperPoint: Self-Supervised Interest Point Detection and Description" bởi DeTone, Malisiewicz và Rabinovich tại Magic Leap, Inc. Bài báo được công bố tại hội nghị CVPR Workshop 2018.

    ### Đặc điểm nổi bật
    - Là một trong những mô hình đầu tiên áp dụng deep learning vào bài toán phát hiện và mô tả đặc trưng
    - Sử dụng phương pháp học tự giám sát (self-supervised learning)
    - Có khả năng hoạt động real-time
    """)
    # Hiển thị hình ảnh kiến trúc mạng
    st.write("### Kiến trúc tổng quan của SuperPoint")
    st.image("UIUX/FeaturesMatching/keke.jpg", caption="Kiến trúc mạng SuperPoint", use_column_width=True)

    st.markdown("""
**1. Encoder (Bộ mã hóa)**
- Là một mạng CNN cơ bản được thiết kế để trích xuất các đặc trưng từ ảnh đầu vào
- Đầu vào là ảnh có kích thước WxHx1 (ảnh xám)
- Sử dụng nhiều lớp tích chập (convolution layers) để giảm kích thước và tăng số kênh đặc trưng
- Kết quả của encoder sẽ được chia sẻ cho cả hai decoder

**2. Interest Point Decoder (Bộ giải mã điểm đặc trưng)**
- Nhận đầu vào từ encoder
- Sử dụng một lớp tích chập để tạo ra feature map có kích thước W/8 x H/8 x 65
- Áp dụng hàm Softmax để chuẩn hóa các giá trị
- Reshape lại để tạo ra bản đồ xác suất các điểm đặc trưng
- Đầu ra là ma trận WxH với giá trị 1 tại vị trí là điểm đặc trưng, 0 ở các vị trí khác

**3. Descriptor Decoder (Bộ giải mã mô tả)**
- Cũng nhận đầu vào từ encoder
- Sử dụng lớp tích chập để tạo ra feature map có kích thước W/8 x H/8 x D
- Áp dụng phép nội suy Bi-Cubic để phóng to kích thước về WxH
- Chuẩn hóa L2 để tạo ra vector mô tả có độ dài D cho mỗi điểm đặc trưng
- Đầu ra là ma trận WxHxD chứa các vector mô tả cho mỗi pixel
""")

    
    # 3. Phần phương pháp đánh giá
    st.header("3. Phương pháp đánh giá")
    st.markdown("""
- Để đánh giá hiệu suất của SuperPoint, SIFT, ORB trên tập dữ liệu Synthetic Shapes Dataset dưới ảnh hưởng của phép quay, thực hiện theo các bước sau:

  - Trích xuất đặc trưng:
    - Trích xuất vector đặc trưng sử dụng SIFT, ORB và SuperPoint tại các keypoint ground truth ở từng góc quay.

- So khớp keypoint:
  - Sử dụng Brute-Force Matching để so khớp các vector đặc trưng đã trích xuất của các keypoint ground truth giữa ảnh gốc và các ảnh đã quay.

- Sử dụng độ đo để tính phần trăm của các keypoint được so khớp chính xác cho mỗi phương pháp ở mỗi góc quay.
""")

  
    
    # 4. Phần kết quả thí nghiệm
    st.header("4. Kết quả thí nghiệm")
    st.write("### Kết quả trên tập kiểm thử")
    st.write("Kết quả matching của SuperPoint với các góc quay khác nhau")

    angle = st.slider("Chọn góc quay", 0, 60, 0, 10)

    base_dir = "UIUX/FeaturesMatching/superpoint_shape_type_images_20241130_094216"
    angle_dir = os.path.join(base_dir, f"angle_{angle}")
    shape_type_dirs = [f"shape_type_{i}" for i in range(8)]

    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    row3_cols = st.columns(2)

    for idx, shape_type_dir in enumerate(shape_type_dirs):
    # Lấy ảnh thứ 2 trực tiếp từ thư mục
        full_dir = os.path.join(angle_dir, shape_type_dir)
        img_path = None
    
        if os.path.exists(full_dir):
            image_files = [f for f in os.listdir(full_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            if len(image_files) >= 2:
                img_path = os.path.join(full_dir, image_files[1])
    
        if img_path:
            if idx < 3:
                col = row1_cols[idx]
            elif idx < 6:
                col = row2_cols[idx - 3]
            else:
                col = row3_cols[idx - 6]
        
            with col:
                try:
                    st.image(img_path, 
                            caption=f"Ảnh kết quả {idx + 1}", 
                            use_column_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi hiển thị ảnh: {str(e)}")

    # Thêm phần hiển thị biểu đồ đánh giá
    st.write("### Kết quả đánh giá")
    st.write("Biểu đồ so sánh độ chính xác (Accuracy) giữa các phương pháp theo góc quay")

    # Hiển thị biểu đồ
    st.image("UIUX/FeaturesMatching/matching.png", 
             caption="Biểu đồ Accuracy theo góc quay", 
             use_column_width=True)

    # Thêm phần giải thích biểu đồ
    st.write("""
    **Nhận xét về kết quả thực nghiệm:**
    Với biểu đồ như trên, một số nhận định được đưa ra về các phương pháp như sau :
    
    - SuperPoint : rất tốt trong khoảng từ 0 đến 40 độ nhưng giảm mạnh trong các góc xoay tiếp theo.
    - ORB : giữ các giá trị khá cao trong khoảng từ 0 đến 40 độ và ở các góc trên 40 độ thì ORB trở nên vượt trội hơn hai phương pháp còn lại.
    - SIFT : không quá nổi trội ở tất cả các góc quay như SuperPoint hay ORB nhưng biên độ dao động nhỏ, có sự ổn định nhất định trên tập dữ liệu.
    """)
    st.markdown("""
    **Có thể lý giải những nhận xét trên như sau :**
                
    - **Đối với SuperPoint :**
      - Là một mô hình học sâu được huấn luyện trên một lượng lớn dữ liệu có chứa các biến đổi nhỏ về góc. Điều này giúp SuperPoint nhận diện và mô tả các đặc trưng rất tốt trong khoảng góc xoay nhỏ (0 đến 40 độ).
      - Nhưng SuperPoint sử dụng mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để trích xuất các đặc trưng. CNN rất mạnh trong việc học các đặc trưng cục bộ nhưng có thể gặp khó khăn khi cần phải tổng quát hóa các biến đổi lớn về không gian như các góc xoay lớn.
      """)
    
    col1,col2,col3  = st.columns([1,7,1])
    with col2 :
      st.image("CNN Architecutre.png",caption="Kiến trúc CNN", use_column_width=True)
    st.markdown("""   
    -
      - Có thể thấy rằng CNN sử dụng các bộ lọc nhỏ (ví dụ 3x3, 5x5) để quét qua hình ảnh và trích xuất các đặc trưng cục bộ. Các bộ lọc này có khả năng phát hiện các mẫu như cạnh, góc, và các cấu trúc hình học nhỏ.
    Và có lớp pooling giúp giảm kích thước không gian của đầu ra sau mỗi lớp tích chập, giữ lại các đặc trưng quan trọng nhưng mất mát một số thông tin không gian chi tiết.
      - Nên khi hình ảnh bị xoay góc lớn các đặc trưng cục bộ có thể di chuyển đến vị trí khác trong hình ảnh, hoặc thay đổi hình dạng, làm cho các bộ lọc tích chập khó nhận diện chính xác.
""")
    
    st.markdown("""
  **Đối với ORB :**
                
    - ORB sử dụng FAST mà vì FAST dựa trên sự thay đổi cường độ giữa các điểm trên vòng tròn và điểm trung tâm, các biến đổi nhỏ về góc thường không thay đổi đáng kể sự so sánh cường độ này.
    - Do đó, các biến đổi nhỏ về góc không ảnh hưởng nhiều đến khả năng phát hiện điểm góc của FAST. Điều này giúp ORB (sử dụng FAST để phát hiện điểm góc) duy trì hiệu suất cao trong các điều kiện biến đổi nhỏ.
    - Còn lý do với các góc lớn ORB vẫn duy trì hiệu suất cao hơn hai phương pháp còn lại là do có khả năng các định hướng đặc trưng.
      - ORB tính toán centroid của vùng xung quanh điểm đặc trưng bằng cách sử dụng Moment. 
      - Tiếp theo, ORB xác định hướng của đặc trưng dựa trên gradient của cường độ pixel trong vùng xung quanh điểm đặc trưng đó. Hướng này được chuẩn hóa về một trục cố định.
      - Hướng của mỗi điểm đặc trưng sẽ được sử dụng để xoay descriptor tương ứng, giúp tạo ra các descriptor không bị ảnh hưởng bởi xoay.
""")
    col1,col2,col3  = st.columns([1,7,1])
    with col2 :
      st.image("467473224_544891085202402_1332492444598727249_n.png",caption="Định hướng đặc trưng", use_column_width=True)
    st.markdown("""
    - Dựa vào hình ảnh, ta có thể thấy rằng :
      - Với góc xoay nhỏ (10°), sự thay đổi không quá lớn nên kết quả trên biểu đồ vẫn cao đáng kể.
      - Với góc xoay lớn (60°), có sự thay đổi đáng kể trong phân bố hướng của các điểm đặc trưng, tuy nhiên, vẫn là yếu tố quan trọn giúp ORB đạt hiệu quả cao hơn các
                phương pháp còn lại dù kết quả vẫn chưa được như ý.
""")
    st.markdown("""
  **Đối với SIFT:**
    - SIFT tạo ra các vector đặc trưng 128 chiều dựa trên histogram gradient, giúp mô tả rất chi tiết các vùng xung quanh điểm đặc trưng.
    - Điều này giúp SIFT rất robust với các biến đổi tỷ lệ và xoay nhờ Gaussian Scale Space và phương pháp xác định hướng đặc trưng nên tính ổn định khá cao.
    - Nhưng SIFT xác định hướng đặc trưng dựa trên histogram gradient cục bộ. Khi gặp các góc quay lớn, histogram có thể bị phân tán hoặc biến dạng, làm giảm độ chính xác của việc xác định hướng.
""")
    col1,col2,col3  = st.columns([1,7,1])
    with col2 :
      st.image("all_results.png", caption="", use_column_width=True)
    st.markdown("""
    - Dựa vào hình ảnh, ta có thể thấy rằng :
      - Sự sai khác giữa các góc xoay không cao tệ nên nhận định SIFT có tính ổn định cao với các góc xoay là khá hợp lí.
      - Tuy nhiên, chất lượng của các góc xoay lại không cao dẫn đến SIFT tuy có độ ổn định cao nhưng chất lượng lại không bằng hai phương pháp kia.
""")
if __name__ == "__main__":
    main()
