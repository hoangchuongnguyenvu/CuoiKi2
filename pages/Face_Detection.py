import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def main():
    st.title("Giới thiệu về Phương pháp Face Detection")

    st.header("1. Giới thiệu về Dataset")
    st.write("""
    Dataset được sử dụng trong nghiên cứu này bao gồm:
    - 800 ảnh kích thước 24x24 pixel, trong đó:
        - 400 ảnh là gương mặt người từ nhiều góc độ và điều kiện ánh sáng khác nhau
        - 400 ảnh không phải gương mặt (negative samples)
    - Dataset được chia thành 2 tập:
        - Tập training: 100% số lượng ảnh (80 ảnh)
        - Tập testing: 50 ảnh 
            + tests/images: chứa ảnh gốc cần test
            + tests/labeled_images: chứa ảnh đã được gán nhãn
            + tests/labels: chứa thông tin nhãn
    """)

    st.write("Một số mẫu từ tập training:")

    SCALE_FACTOR = 4  # Hệ số phóng đại, ví dụ 24*4 = 96 pixels
    DISPLAY_SIZE = 24 * SCALE_FACTOR  # Kích thước hiển thị cuối cùng

    # Hiển thị ảnh gương mặt với kích thước phóng đại đồng nhất
    st.write("Ảnh gương mặt:")
    face_cols = st.columns(6)
    for i in range(6):
        with face_cols[i]:
            try:
                # Đọc ảnh và resize với hệ số phóng đại
                image = Image.open(f"UIUX/FaceDetection/dataset/face/{i+1}.png")
                image = image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
                st.image(image, 
                        caption=f"Face {i+1}",
                        width=DISPLAY_SIZE)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh face/{i+1}.png")

    # Hiển thị ảnh không phải gương mặt với kích thước phóng đại đồng nhất
    st.write("Ảnh không phải gương mặt:")
    nonface_cols = st.columns(6)
    for i in range(6):
        with nonface_cols[i]:
            try:
                # Đọc ảnh và resize với hệ số phóng đại
                image = Image.open(f"UIUX/FaceDetection/dataset/nonface/{i+1}.png")
                image = image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
                st.image(image, 
                        caption=f"Non-face {i+1}",
                        width=DISPLAY_SIZE)
            except FileNotFoundError:
                st.error(f"Không tìm thấy ảnh nonface/{i+1}.png")

    st.write("Một số mẫu từ tập test:")
    
    # Khởi tạo list đường dẫn ảnh
    image_paths = [
        "UIUX/FaceDetection/tests/labeled_images/000020.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000044.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000136.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000191.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000236.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000405.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000495.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000590.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000698.jpg",
        "UIUX/FaceDetection/tests/labeled_images/000760.jpg"
    ]
    
    # Hàng 1
    test_row1 = st.columns(5)
    for i in range(5):
        with test_row1[i]:
            st.image(image_paths[i], caption=f"Test {i}")

    # Hàng 2
    test_row2 = st.columns(5)
    for i in range(5):
        with test_row2[i]:
            st.image(image_paths[i+4], caption=f"Ground Truth {i+4}")

    st.header("2. Giới thiệu Phương pháp")
    st.markdown("""
    Phương pháp Face Detection sử dụng mô hình Cascade Classifier kết hợp với kNN, với quy trình bao gồm 2 phần chính:

    - Quá trình training:
        - **(1)** Đầu vào là một tập dữ liệu bao gồm ảnh gương mặt và ảnh không phải gương mặt được chuyển sang chế độ Integral
        - **(2)** Trích xuất một số lượng lớn đặc trưng từ tập dữ liệu
        - **(3)** Đưa qua Adaboost để chọn ra các đặc trưng hữu ích
        - **(4)** Có được các tập đặc trưng hữu ích phục vụ cho quá trình phát hiện  """)
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('UIUX/FaceDetection/metric_process/1.png', caption='Mô tả quá trình huấn luyện')

    st.markdown("""
    - Phương pháp phát hiện gương mặt (Face Detection):
        - **(1)** Ảnh dầu vào được chia thành nhiều cửa sổ nhỏ **(subwindows)** để quét toàn bộ hình ảnh.
        - **(2)** Các cửa sổ này sẽ được duyệt tuần tự để kiểm tra sự hiện diện của khuôn mặt.
        - **(3)** Với từng cửa sổ con, đặc trưng quan trọng sẽ được trích xuất.
        - **(4)** Sau đó, các đặc trưng của từng cửa sổ con được so sánh với vector đặc trưng của tập huấn luyện.
        - **(5)** Sử dụng thuật toán **K-Nearest Neighbors (KNN)** để phân loại các cửa sổ con.
        - **(6)** Nếu cửa sổ con được xác định là gương mặt, thì sẽ được phân loại là gương mặt, còn không thì ngược lại
    """)

    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('UIUX/FaceDetection/metric_process/2.png', caption='Mô tả quá trình phát hiện gương mặt trên ảnh bất kì')

    st.header("3. Tham số huấn luyện")
    st.write("""
    Các tham số chính trong quá trình huấn luyện mô hình kết hợp:

    1. **Tham số dữ liệu**:
        - `numPos`: 400 mẫu 
        - `numNeg`: 400 mẫu 
        - `numStages`: 6 stages
        - `w`, `h`: 24x24 pixels

    2. **Tham số Cascade**:
        - `featureType`: `BASIC` (sử dụng các Haar features cơ bản)
        - `bt`: `GAB` (Gentle AdaBoost)
        - `minHitRate`: 0.995 (tỷ lệ hit tối thiểu cho mỗi stage)
        - `maxFalseAlarmRate`: 0.5 (tỷ lệ false alarm tối đa cho mỗi stage)
        - `numFeatures`: 16 features được sử dụng

    3. **Tham số kNN**:
        - `k`: chạy từ 1 đến 31 (step = 2)
        - `distance_metric`: `euclidean` (khoảng cách Euclidean)
        - `weights`: `distance` (trọng số theo khoảng cách)

    **Kết quả cấu hình cascade:**
    - Kích thước cửa sổ huấn luyện: 24x24 pixels
    - Số stages trong cascade: 6 stages
    - Số features được sử dụng: 16 features
    - Cấu trúc cascade: Phân tầng với 6 stages xếp tầng, mỗi stage sử dụng một tập con của 16 features
        + Stage 0: 3 features
        + Stage 1: 3 features  
        + Stage 2: 3 features
        + Stage 3: 3 features
        + Stage 4: 3 features
        + Stage 5: 1 feature
    """)

    st.header("4. Phương pháp Đánh giá")
    st.write("""
    Để đánh giá hiệu quả của thuật toán Watershed trong việc phân đoạn biển số xe, nghiên cứu sử dụng metric Intersection over Union (IoU). Đây là một trong những phương pháp đánh giá phổ biến nhất trong các bài toán phân đoạn ảnh.
    IoU được tính bằng tỷ số giữa diện tích phần giao (Intersection) và diện tích phần hợp (Union) của hai vùng:
    - Vùng được phân đoạn bởi thuật toán (Predicted region)
    - Vùng ground truth (Ground Truth)
    
    Đặc điểm:
    - Giá trị ∈ [0, 1]
    - IoU = 1: phân đoạn hoàn hảo
    - IoU > 0.5: kết quả chấp nhận được
    """)

    col1, col2, col3 = st.columns([1,4,1])  # Tỷ lệ 1:2:1
    with col2:  # Cột giữa
        st.image("UIUX/FaceDetection/metric_process/OIP.jpg", 
                caption="Minh họa cách tính IoU giữa Predicted Box và Ground Truth Box")

    st.header("5. Kết quả Huấn luyện và Test")

    # Phần 1: Kết quả huấn luyện
    st.subheader("5.1 Kết quả Huấn luyện")
    
    # Hàng 1: Biểu đồ
    st.image("UIUX/FaceDetection/FD.png", caption="Biểu đồ hiệu suất KNN với Haar Features")

    # Hàng 2: Phân tích chi tiết
    st.write("""
    **Kết quả phân tích:**
    - Giá trị k tốt nhất: k = 13
    - IoU tương ứng: 0.2158 (≈ 0.216)
    """)

    # Phần 2: Kết quả test
    st.subheader("5.2 Kết quả Test trên Tập Test")
    
    # Hàng 1: Ảnh test gốc
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5, row1_col6, row1_col7, row1_col8, row1_col9, row1_col10 = st.columns(10)

    with row1_col1:
        st.image("UIUX/FaceDetection/tests/images/000020.jpg", caption="Ảnh test 1")
    with row1_col2:
        st.image("UIUX/FaceDetection/tests/images/000044.jpg", caption="Ảnh test 2")
    with row1_col3:
        st.image("UIUX/FaceDetection/tests/images/000136.jpg", caption="Ảnh test 3")
    with row1_col4:
        st.image("UIUX/FaceDetection/tests/images/000191.jpg", caption="Ảnh test 4")
    with row1_col5:
        st.image("UIUX/FaceDetection/tests/images/000236.jpg", caption="Ảnh test 5")
    with row1_col6:
        st.image("UIUX/FaceDetection/tests/images/000405.jpg", caption="Ảnh test 6")
    with row1_col7:
        st.image("UIUX/FaceDetection/tests/images/000495.jpg", caption="Ảnh test 7")
    with row1_col8:
        st.image("UIUX/FaceDetection/tests/images/000590.jpg", caption="Ảnh test 8")
    with row1_col9:
        st.image("UIUX/FaceDetection/tests/images/000698.jpg", caption="Ảnh test 9")
    with row1_col10:
        st.image("UIUX/FaceDetection/tests/images/000760.jpg", caption="Ảnh test 10")

    # Hàng 2: Kết quả sau khi áp dụng mô hình
    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5, row2_col6, row2_col7, row2_col8, row2_col9, row2_col10 = st.columns(10)

    with row2_col1:
        st.image("UIUX/FaceDetection/tests/result/detected_000020.jpg", caption="Kết quả 1")
    with row2_col2:
        st.image("UIUX/FaceDetection/tests/result/detected_000044.jpg", caption="Kết quả 2")
    with row2_col3:
        st.image("UIUX/FaceDetection/tests/result/detected_000136.jpg", caption="Kết quả 3")
    with row2_col4:
        st.image("UIUX/FaceDetection/tests/result/detected_000191.jpg", caption="Kết quả 4")
    with row2_col5:
        st.image("UIUX/FaceDetection/tests/result/detected_000236.jpg", caption="Kết quả 5")
    with row2_col6:
        st.image("UIUX/FaceDetection/tests/result/detected_000405.jpg", caption="Kết quả 6")
    with row2_col7:
        st.image("UIUX/FaceDetection/tests/result/detected_000495.jpg", caption="Kết quả 7")
    with row2_col8:
        st.image("UIUX/FaceDetection/tests/result/detected_000590.jpg", caption="Kết quả 8")
    with row2_col9:
        st.image("UIUX/FaceDetection/tests/result/detected_000698.jpg", caption="Kết quả 9")
    with row2_col10:
        st.image("UIUX/FaceDetection/tests/result/detected_000760.jpg", caption="Kết quả 10")


    st.header("6. Ứng dụng Thử nghiệm")
    st.write("""
    Dưới đây là demo ứng dụng phát hiện khuôn mặt. Bạn có thể tải lên một hình ảnh để thử nghiệm:
    - Hỗ trợ các định dạng: JPG, PNG, JPEG
    - Có thể phát hiện nhiều khuôn mặt trong một ảnh
    - Kết quả hiển thị bounding box màu xanh quanh khuôn mặt
    """)
    
    def apply_face_detection(image):
        # Chuyển đổi sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier('UIUX/FaceDetection/output/cascade.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(24, 24)
        )
        
        # Vẽ rectangle xung quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return image, len(faces)  # Trả về cả ảnh v số khuôn mặt phát hiện được

    # Phần upload và xử lý ảnh
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện khuôn mặt", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Hiển thị ảnh gốc và ảnh sau khi xử lý
        col1, col2 = st.columns(2)
        
        # Đọc và hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        with col1:
            st.write("**Ảnh gốc:**")
            st.image(image, use_column_width=True)
        
        # Xử lý ảnh
        img_array = np.array(image)
        processed_img, num_faces = apply_face_detection(img_array)
        
        # Hiển thị ảnh đã xử lý
        with col2:
            st.write("**Ảnh sau khi nhận diện:**")
            st.image(processed_img, use_column_width=True)
            st.write(f"*Số khuôn mặt phát hiện được: {num_faces}*")
        
        # Thêm nút tải ảnh đã xử lý
        processed_img_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        processed_img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Tải ảnh đã xử lý",
            data=byte_im,
            file_name="face_detected.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
