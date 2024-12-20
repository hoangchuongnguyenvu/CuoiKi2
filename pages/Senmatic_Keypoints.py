import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2

def main():
    st.title("Phân tích và Đánh giá Phương pháp Phát hiện Keypoint")

    # Phần 1: Giới thiệu Dataset
    st.header("1. Giới thiệu Dataset")
    st.write("""
    Dataset synthetic được tạo ra để đánh giá hiệu quả của các thuật toán phát hiện góc. Dataset bao gồm các thư mục con sau:

    1. **draw_checkerboard**: Chứa các hình ảnh bàn cờ vua với các ô đen trắng xen kẽ
    2. **draw_cube**: Chứa các hình ảnh khối lập phương 3D
    3. **draw_ellipses**: Chứa các hình ảnh hình elip
    4. **draw_lines**: Chứa các hình ảnh đường thẳng
    5. **draw_multiple_polygons**: Chứa các hình ảnh nhiều đa giác
    6. **draw_polygon**: Chứa các hình ảnh đa giác đơn
    7. **draw_star**: Chứa các hình ảnh hình ngôi sao
    8. **draw_stripes**: Chứa các hình ảnh các dải sọc
    9. **gaussian_noise**: Chứa các hình ảnh nhiễu Gaussian

    Mỗi thư mục con chứa các hình ảnh được tạo ra với các đặc điểm khác nhau, giúp đánh giá khả năng phát hiện góc của thuật toán trong nhiều trường hợp khác nhau.
    """)

    # Danh sách tên file ảnh thực tế
    original_images = [
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_checkerboard_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_cube_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_lines_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_multiple_polygons_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_polygon_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_star_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_stripes_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/gaussian_noise_original.png",
        "application/Senmatic_Keypoints/detector_results5/original_images/draw_ellipses_original.png"
        
    ]

    gt_images = [
        "application/Senmatic_Keypoints/detector_results5/draw_checkerboard_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_cube_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_lines_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_multiple_polygons_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_polygon_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_star_result.png",
        "application/Senmatic_Keypoints/detector_results5/draw_stripes_result.png"
    ]

    # Style cho container
    st.markdown("""
        <style>
            .stImage > img {
                max-width: 100%;
                height: auto;
            }
            .image-container {
                padding: 10px;
            }
            .center-row {
                display: flex;
                justify-content: center;
                gap: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc:")

# Hàng 1: Hiển thị 5 ảnh đầu
    row1 = st.columns(5)
    for i, img_path in enumerate(original_images[:5]):  # Lấy 5 ảnh đầu tiên
        with row1[i]:
            st.image(img_path, caption=f"Ảnh {i+1}")

# Hàng 2: Hiển thị 4 ảnh còn lại
    row2 = st.columns(4)
    for i, img_path in enumerate(original_images[5:]):  # Lấy 4 ảnh còn lại
        with row2[i]:
            st.image(img_path, caption=f"Ảnh {i+6}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Ground Truth với cùng layout
    st.subheader("Ground Truth:")
    # Hàng 1: 4 ảnh đầu
    row1_col2 = st.columns(5)
    for i, gt_name in enumerate(gt_images[:5]):
        with row1_col2[i]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(gt_name, caption=f"Ground Truth {i+1}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Hàng 2: 3 ảnh còn lại, căn giữa
    st.markdown('<div class="center-row">', unsafe_allow_html=True)
    row2_col2 = st.columns(4)
    for i, gt_name in enumerate(gt_images[5:]):
        with row2_col2[i+1]:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(gt_name, caption=f"Ground Truth {i+5}")
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Phần 2: Phương pháp
    st.header("2. Giới thiệu phương pháp")
    st.subheader("2.1 Scale Invariant Feature Transform (SIFT)")
    # SIFT

    st.markdown("""
1. **Tạo Không Gian Gaussian**:
   - Bắt đầu từ hình ảnh gốc, tạo ra các hình ảnh ở nhiều độ phân giải khác nhau bằng cách làm mờ dần.
   - **Chi tiết**: Sử dụng bộ lọc Gaussian để làm mờ hình ảnh ở nhiều tỷ lệ khác nhau. Điều này giúp phát hiện các đặc trưng trên nhiều cấp độ chi tiết.

2. **Tính Toán Difference of Gaussian (DoG)**:
   - Tạo ra các hình ảnh DoG bằng cách trừ các hình ảnh Gaussian liền kề.
   - **Chi tiết**: Sự khác biệt giữa các hình ảnh Gaussian ở các mức độ khác nhau giúp xác định các điểm đặc trưng cục bộ. Đây là bước quan trọng để phát hiện các cạnh và điểm đặc trưng mạnh.

3. **Phát Hiện Các Điểm Đặc Trưng**:
   - Xác định các điểm đặc trưng ổn định từ hình ảnh DoG.
   - **Chi tiết**: So sánh mỗi pixel với các điểm lân cận trong các mức độ hiện tại và liền kề để xác định các điểm đặc trưng. Điểm đặc trưng là những điểm có giá trị cao nhất hoặc thấp nhất so với các điểm xung quanh.

4. **Tính Toán Gradient và Hướng**:
   - Tính toán độ lớn và hướng của gradient cho mỗi điểm đặc trưng.
   - **Chi tiết**: Độ lớn và hướng của gradient được tính toán bằng cách đo sự thay đổi cường độ pixel xung quanh điểm đặc trưng. Các giá trị gradient này giúp mô tả chi tiết hình ảnh cục bộ.

5. **Tạo Vector Đặc Trưng SIFT**:
   - Sử dụng các giá trị gradient để tạo vector đặc trưng 128 chiều.
   - **Chi tiết**: Các giá trị gradient xung quanh mỗi điểm đặc trưng được lập thành biểu đồ và sau đó được sử dụng để xây dựng một vector đặc trưng 128 chiều. Vector này mô tả duy nhất vùng hình ảnh cục bộ xung quanh điểm đặc trưng, phục vụ cho việc nhận dạng và khớp hình ảnh.
""")


    # Vị trí để thêm hình ảnh SIFT
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("application/Senmatic_Keypoints/ORB_process/SIFT.png", caption="SIFT Keypoints Detection")

    # ORB
    st.subheader("2.2 Oriented FAST and Rotated BRIEF (ORB)")
    st.markdown("""
1. **Hình Ảnh Đầu Vào**:
   - Bắt đầu từ các hình ảnh đầu vào cần phân tích.

2. **Tìm Các Điểm Đặc Trưng Bằng FAST**:
   - Sử dụng thuật toán FAST (Features from Accelerated Segment Test) để tìm các điểm đặc trưng.
   - **Chi tiết**: Thuật toán FAST rất nhanh trong việc phát hiện các điểm đặc trưng bằng cách kiểm tra sự thay đổi cường độ pixel xung quanh mỗi pixel.

3. **Chọn Các Điểm Tốt Nhất Bằng Harris**:
   - Sử dụng phương pháp Harris Corner Detection để chọn ra các điểm đặc trưng tốt nhất từ những điểm đã được tìm thấy bởi FAST.
   - **Chi tiết**: Harris Corner Detection giúp xác định các điểm góc tốt nhất dựa trên mức độ thay đổi cường độ pixel, giúp cải thiện độ chính xác của đặc trưng được chọn.

4. **Trích Xuất Descriptor Nhị Phân Bằng BRIEF**:
   - Sử dụng thuật toán BRIEF (Binary Robust Independent Elementary Features) để trích xuất các descriptor nhị phân cho các điểm đặc trưng đã được chọn.
   - **Chi tiết**: BRIEF tạo ra các descriptor nhị phân dựa trên sự so sánh cường độ pixel trong một vùng xung quanh điểm đặc trưng, giúp giảm độ phức tạp tính toán và tăng tốc độ so khớp đặc trưng.
                
5. **Hình Ảnh Kết Quả**:
   - Xuất ra các hình ảnh đã được phát hiện và mô tả các đặc trưng.
   - **Chi tiết**: Hình ảnh đầu ra chứa các điểm đặc trưng đã được phát hiện và mô tả, sẵn sàng cho các tác vụ thị giác máy tính khác như nhận dạng đối tượng, ghép nối hình ảnh.
""")

    # Vị trí để thêm hình ảnh ORB
    col1, col2, col3 = st.columns([1,7,1])
    with col2:
        st.image("ORBhehe.jpg", caption="ORB Feature Detection")

    # Euclidean Distance
    st.subheader("2.3 Euclidean Distance")
    st.write("""
    Trong context của đánh giá độ chính xác của việc phát hiện điểm đặc trưng, khoảng cách Euclidean được sử dụng để xác định xem một điểm được phát hiện có thực sự gần với groundtruth hay không.

    **Nguyên lý hoạt động:**
    1. **Vòng tròn Euclidean:**
       - Với mỗi điểm groundtruth, vẽ một vòng tròn với bán kính r
       - Bán kính r là ngưỡng khoảng cách Euclidean được chấp nhận
       - Tạo ra một vùng chấp nhận xung quanh mỗi điểm groundtruth

    2. **Đánh giá độ chính xác:**
       - Một điểm được phát hiện (detected point) được coi là đúng nếu nó nằm trong vòng tròn
       - Công thức tính khoảng cách: d = √[(x₁-x₂)² + (y₁-y₂)²]
       - Nếu d ≤ r: điểm phát hiện được coi là chính xác
       - Nếu d > r: điểm phát hiện được coi là sai

    **Ưu điểm của phương pháp:**
    - Cho phép một độ sai số chấp nhận được
    - Đơn giản và trực quan trong việc đánh giá
    - Phù hợp với nhiều loại dataset khác nhau

    **Các tham số quan trọng:**
    - Bán kính r: quyết định độ nghiêm ngặt của việc đánh giá
    - Càng nhỏ r: đánh giá càng nghiêm ngặt
    - Càng lớn r: cho phép sai số nhiều hơn
    """)

    # Vị trí để thêm hình ảnh Euclidean Distance
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("application/Senmatic_Keypoints/results/euclidean_visualization.png", caption="Euclidean Distance Evaluation")

    # 2.2 Kết quả thực nghiệm
    st.header("3. Kết quả thực nghiệm")
    
    # Kết quả ORB
    st.markdown("#### Kết quả ORB")
    orb_files = sorted([f for f in os.listdir("application/Senmatic_Keypoints/ORB") if f.endswith(('.jpg', '.png', '.jpeg'))])
    cols_orb = st.columns(4)
    for i, image_file in enumerate(orb_files[:20]):  # Giới hạn 16 ảnh
        with cols_orb[i % 4]:
            image_path = os.path.join("application/Senmatic_Keypoints/ORB", image_file)
            st.image(image_path, caption=f"ORB {i+1}")

    # Kết quả SIFT
    st.markdown("#### Kết quả SIFT")
    sift_files = sorted([f for f in os.listdir("application/Senmatic_Keypoints/SIFT") if f.endswith(('.jpg', '.png', '.jpeg'))])
    cols_sift = st.columns(4)
    for i, image_file in enumerate(sift_files[:20]):  # Giới hạn 16 ảnh
        with cols_sift[i % 4]:
            image_path = os.path.join("application/Senmatic_Keypoints/SIFT", image_file)
            st.image(image_path, caption=f"SIFT {i+1}")

    # Phần 3: Phương pháp đánh giá
    st.header("4. Phương pháp đánh giá")

    # Phần giải thích Precision và Recall
    st.write("""
    **Precision (Độ chính xác)**
    Precision = TP / (TP + FP)
    - TP (True Positive): Số keypoint phát hiện đúng
    - FP (False Positive): Số keypoint phát hiện sai

    **Recall (Độ phủ)**
    Recall = TP / (TP + FN)
    - TP (True Positive): Số keypoint phát hiện đúng
    - FN (False Negative): Số keypoint bỏ sót
    """)

    # Thêm hình minh họa ở cuối
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image("application/Senmatic_Keypoints/ORB_process/R.png", caption="Minh họa Precision và Recall")

    # Phần 4: Kết quả đánh giá
    st.header("5. Kết quả đánh giá")
    st.subheader("5.1 Biểu đồ đánh giá")
    col1, col2, col3 = st.columns([1,20,1])
    with col2:
        st.image("laluot.png", 
                 caption="Biểu đồ so sánh kết quả đánh giá SIFT và ORB", 
                 use_column_width=True)

    # Phần 5: Thảo luận
    st.subheader("5.2 Thảo luận")
    st.markdown("""
**Nhận xét chung**
                
- SIFT chỉ cạnh tranh ngang ngửa với ORB ở hình **Lines** và thể hiện tốt hơn ở **Stripes**.
- ORB cho kết quả tốt hơn SIFT ở hầu hết các hình dạng, đặc biệt nổi bật hơn cả là ở các hình dạng như **Checkerboard**, **Polygon**, và **Star**.
""")

    st.markdown("**Dạng hình học nổi bật của SIFT**")
    col1 , col2 = st.columns(2)
    with col1:
        st.image("kaggle_working_20241215_112046/draw_lines_comparison.png", caption="Lines")
    with col2 :
        st.image("kaggle_working_20241215_112046/draw_stripes_comparison.png",caption="Stripes")

    st.markdown("**Dạng hình học nổi bật của ORB**")
    col1 , col2 , col3 = st.columns(3)
    with col1:
        st.image("kaggle_working_20241215_112046/draw_checkerboard_comparison.png", caption="Checkerboard")
    with col2 :
        st.image("kaggle_working_20241215_112046/draw_polygon_comparison.png",caption="Polygon") 
    with col3 : 
        st.image("kaggle_working_20241215_112046/draw_star_comparison.png",caption="Star") 
    
    st.markdown("""
**Giải thích nhận xét**
- Từ kết quả thực tế, ta có được các nhận xét chung kể trên. Hay nói một cách khác thì các nhận xét đó cho ta được môt nhận
xét mới về SIFT và ORB như sau :
    - SIFT sẽ mạnh trong các hình dạng có ít góc sắc nét (hạn chế hoặc không xuất hiện các góc).
    - ORB sẽ vượt trội trong các dữ liệu mà ở đó dữ liệu có các hình dạng đơn giản (tam giác hoặc tứ giác) và có nhiều góc rõ ràng.
                                
- Thật vậy, có thể xem thông quá cơ chế hoạt động của hai thuật toán để lý giải vấn đề trên như sau :
    - Đối với SIFT, dựa trên gradient cường độ pixel nên không phụ thuộc vào góc sắc nét thì việc sử dụng một 
    descriptor 128 chiều sẽ giúp cho các đặc trưng được phát hiện chi tiết và chính xác hơn.
""")
    col1,col2,col3 = st.columns([1,7,1])
    with col2 :
        st.image("feature_detection_analysis.png", caption="Minh họa")    

    st.markdown("""
- Bạn có thể nhìn ở hình ảnh so sánh ở trên của hai phương pháp thì có thể thấy được rằng :
                
    - SIFT có Gaussian Scale Space để làm mờ ảnh với nhiều tỉ lệ khác nhau để làm bật lên những chi tiết lớn và làm mờ những chi tiết nhỏ 
    khiến cho đặc trưng quan trọng có thể được phát hiện ở bất kỳ mức độ chi tiết nào, từ chi tiết nhỏ đến chi tiết lớn, mà không bị ảnh hưởng bởi sự thay đổi về kích thước hay góc sắc nét của hình ảnh gốc.
    - Đồng thời, DoG giúp làm nổi bật các cạnh và các đặc trưng chính trong hình ảnh bằng cách khi hai hình ảnh Gaussian bị trừ đi nhau, các thay đổi nhanh về cường độ 
    (các cạnh và góc) sẽ được làm nổi bật. Điều này có nghĩa là DoG sẽ nhấn mạnh các vùng có thay đổi cường độ lớn, giúp phát hiện các đặc trưng mà không phụ thuộc vào hình dạng góc cạnh của hình ảnh ban đầu. 
    Việc này làm cho SIFT trở nên hiệu quả trong việc phát hiện các đặc trưng ở các hình ảnh mà các góc sắc nét không quan trọng.
    - Kết hợp với việc có một descriptor 128 chiều làm cho các thông tin về đặc trưng rất chi tiết và chính xác hơn rất nhiều.
                
- Còn ở phía ngược lại, trên hình ảnh, ORB cũng phát hiện ra rất nhiều đặc trưng nhưng tỉ lệ sai lệch là rất lớn. Đặc biệt là nằm rất nhiều trên chính đường thẳng 
. Nguyên nhân chính là do :
                
    - ORB sử dụng FAST (Features from Accelerated Segment Test) để phát hiện các keypoints nhưng phương pháp này lại bị ảnh hưởng rất nhiều từ việc phải xuất hiện các góc sắc nét.
    Do không giống như SIFT - làm bật lên các đặc trưng thông qua gradient cường độ bằng cách làm mờ và đánh giá. Còn ORB lại quan tâm đến sự thay đổi đột ngột 
    của gradient cường độ để phát hiện ra góc.
    - Do đó, đối với các đường thẳng - nơi mà các gradient cường độ biến thiên đều đặn thì ORB lại dễ dàng phát hiện ra các điểm keypoints sai lệch.
    - Tuy vẫn có bộ lọc các keypoint yếu bằng Harris Corner Measure sau khi sử dụng FAST để phát hiện. Nhưng, phương pháp nào cũng có những điểm yếu.
    Việc tập trung vào chỉ số "cornerness"  để đánh giá đó là góc vẫn chưa đủ toàn diện. Nếu những keypoint có reponse yếu nào vẫn đủ sức vượt qua
    ngưỡng tối thiểu của chỉ số "cornerness" thì vẫn được tính là một keypoint.
""")
    st.markdown("""
- Đối với ORB :       
    - Còn đối với ORB, việc sử dụng FAST (Features from Accelerated Segment Test) là yếu tố phát hiện tốt các góc trong hình ảnh. Nhất
    là với các hình ảnh có góc sắc nét. Nơi mà có sự giao nhau giữa nhiều đường thẳng hay cạnh trong đa giác.
""")
    
    col1,col2,col3 = st.columns([1,7,1])
    with col2 :
        st.image("feature_detection_analysis (2).png", caption="Minh họa")

    col1,col2,col3 = st.columns([1,7,1])
    with col2 :
        st.image("groundtruth_comparison (2).png", caption="Minh họa")
        
    st.markdown("""
- Thông qua hai hình ảnh so sánh ở trên, ta có thể đưa ra lí giải cho những nhận xét ở trên như sau :
    - Đầu tiên, ta có thể thấy được SIFT phát hiện sai toàn bộ với chỉ số Recall và Precision rất thấp. Nguyên nhân là do :
        - Mặc dù, SIFT sử dụng Gaussian Scale Space và DoG để phát hiện các đặc trưng phức tạp và ổn định. Nên trong một số hình ảnh đa giác đơn giản
        thì các đặc trưng này có thể không đủ rõ ràng hoặc phức tạp để SIFT phát hiện.
        - Nếu sự thay đổi cường độ không đủ mạnh hoặc không ổn định, DoG có thể bỏ qua hoặc phát hiện sai điểm cực trị. Điều này dẫn đến việc SIFT không phát hiện đúng các đặc trưng quan trọng trong hình ảnh.
""")
    col1,col2,col3 = st.columns([1,5,1])
    with col2 :
        st.image("Figure2-aConstruction-of-the-Difference-of-GaussianDoG-scale-space-b-similarity.png", caption="Cực trị trong DoG")

    st.markdown("""
- Còn đối với ORB thì như chúng ta đã biết thì phương pháp rất mạnh trong việc phát hiện các góc thông qua nguyên lí hoạt động của nó.
""")
    col1,col2,col3 = st.columns([1,5,1])
    with col2 :
        st.image("Example-of-the-FAST-features-from-accelerated-segment-test-feature-detector.png", caption="Nguyên lí hoạt động của FAST")

    st.markdown("""
- 
    - FAST phát hiện điểm góc bằng cách kiểm tra cường độ của 16 pixel xung quanh pixel trung tâm và xác định nó là điểm góc nếu có ít nhất 9 pixel sáng hơn hoặc tối hơn một ngưỡng xác định.

    - FAST rất mạnh trong việc phát hiện các góc sắc nét, nơi sự thay đổi cường độ lớn và đột ngột.
""")
if __name__ == '__main__':
    main()
