import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def main():
    st.title("KCF (Kernelized Correlation Filter) Object Tracking")

    # Phần 1
    st.header("Phần 1: Nguyên lý hoạt động")
    
    st.markdown("""
    ### A. Tổng quan
    **KCF** (**Kernelized Correlation Filter**) là thuật toán **tracking** dựa trên **correlation filter** với **kernel trick**. Điểm đặc biệt của **KCF** là sử dụng tính chất tuần hoàn của ma trận để tăng tốc độ tính toán và áp dụng **kernel trick** để xử lý trong không gian đặc trưng phi tuyến.
    """)
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('4.png', caption='Mô tả quá trình huấn luyện')

    st.markdown("""
    Mô tả :
    - **(1)** : Sau khi chọn ROI từ frame đầu tiên thì sẽ xử lí ROI về dạng gray scale và tạo target.
    - **(2)** : Từ đó, ta thực hiện tính toán gradient theo 2 hướng (x,y).
    - **(3)** : Nhờ vào hai ma trận gradient theo hai hướng đó thì ta có thể tạo ra được :
        - Tính magnitude - Thể hiện mức độ thay đổi cường độ tại mỗi pixel
        - Tính orientation - Thể hiện hướng thay đổi cường độ (Giá trị từ 0 đến 180 độ).
    - **(4)** : Thông qua bước trên có thể trích xuất được HOG (Histogram of Oriented Gradients) features - mô tả phân bố gradient trong vùng ảnh
    - **(5)** : Để tăng tốc độ tính toán và giảm chi phí tính toán thì đưa HOG features mới trích xuất được qua FFT.
    - **(6)** : Đưa vào bộ lọc KCF để huấn luyện.
    - **(7)** : Tạo ra một mẫu mục tiêu lý tưởng để huấn luyện bộ lọc KCF hay nói cách khác, định nghĩa "hình dạng" mà bộ lọc nên tìm kiếm khi theo dõi đối tượng.
    - **(8)** : Frame tiếp theo được nới rộng Search Area, sau đó đưa qua bộ lọc KCF để tìm kiếm đối tượng.
    - **(9)** : Tạo ra reponse map - vị trị như này sẽ được coi là vị trí mới của đối tượng .
    - **(10)**  : Từ đó sẽ khoanh vùng đối tượng cho frame tiếp theo .
    - **(11)**  : Để thực hiện ở các frame tiếp theo thì lấy ra tiếp HOG features của frame đó và learning rate .
    - **(12)**  : Đưa qua FFT rồi quay lại bước **(6)** và thực hiện cho đến khi kết thúc
    #### B. Các tham số trong implementation:
    1. **Tham số video:**
    - Frame Width: `frame_width = 640`
    - Frame Height: `frame_height = 480`
    - FPS: `fps = 30`
    - Video Format: `fourcc = 'mp4v'`

    2. **Tham số hiển thị:**
    - Tracking Color: `(0, 255, 0)` (`Green`)
    - Lost Color: `(0, 0, 255)` (`Red`)
    - Line Thickness: `2`
    - Font: `cv2.FONT_HERSHEY_SIMPLEX`
    - Font Scale: `1.0`

    """)

    # Giới thiệu chung cho cả 2 video
    st.markdown("""
    ### C. Ví dụ minh họa
    Video dưới đây thể hiện khả năng theo dõi chuyển động của KCF:
    - Đối tượng: Người đi bộ trên đường
    - Điều kiện: Nền đơn giản, ánh sáng tốt
    """)

    # Hiển thị 2 video trong 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Video 1**")
        video_file = open('UIUX/KCF/output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with col2:
        st.markdown("**Video 2**")
        video_file = open('UIUX/KCF/output1.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    # Chuyển sang Phần 3
    st.header("Phần 2: Thảo luận về các trường hợp thách thức")
    tabs = st.tabs(["Background Clutters"])

    with tabs[0]:
        st.subheader("Background Clutters (Nền phức tạp)")
        # Tạo 2 cột để hiển thị video
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video gốc**")
            video_path = 'UIUX/KCF/walking_output.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                # Debug info
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        with col2:
            st.markdown("**Video có áp dụng KCF tracking**")
            video_path = 'UIUX/KCF/walking_result.mp4'
            if os.path.exists(video_path):
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"Không tìm thấy file: {video_path}")
                # Debug info
                st.write("Current directory:", os.getcwd())
                st.write("Files in videos folder:", os.listdir('videos'))

        # Phần nhận xét khả năng tracking
        st.markdown("""
    **Tổng Quan**

    Báo cáo này phân tích chi tiết quá trình mất tracking trong hệ thống theo dõi đối tượng KCF (Kernelized Correlation Filter).
    Dựa trên dữ liệu từ 4 frame liên tiếp trước và tại thởi điểm mất tracking, với sự phát hiện nhiều thay đổi đáng chú ý
    trong đặc trưng HOG (Histogram of Oriented Gradients).
    """)

    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('anh1.png', caption='Bốn frame liên tiếp được dùng để đánh giá')
    
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('anh5.png', caption='Bốn ma trận HOG liên tiếp được dùng để đánh giá')



    # Phân tích thống kê
    st.markdown("""
    **Phân Tích Thống Kê**

    Với từng gam màu có trong ma trận HOG kể trên thì ta có thể thống kê thành các giá trị trung bình để đánh giá sự sai khác và thay đổi trong quá trình lấy đặc trựng 
    mạnh ở ma trận HOG. Với việc lấy Frame -3 làm mốc thì ta nhận thấy :
    """)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        | Frame | Mean Value | Thay Đổi |
        |-------|------------|-----------|
        | Frame -3 | 7041.62 | Baseline |
        | Frame -2 | 7281.82 | ⬆️ +3.4% |
        | Frame -1 | 7097.88 | ⬇️ -2.5% |
        | Frame at loss | 6617.20 | ⬇️ -6.8% |
        """)
    st.markdown("""
    *Nhận xét*: 
    
    - Sự suy giảm liên tục của giá trị trung bình cho thấy đặc trưng của đối tượng đang dần thay đổi.
    - Tuy chúng ta biết rằng dù thay đổi về ánh sáng dù là nhỏ nhất trên ma trận HOG cũng làm biến đổi nhiều giá trị
    nhưng để đánh giá một cách tốt hơn thì ta cần phân tích chi tiết các thay đổi có trong ma trận.
    """)

    # Phân tích chi tiết
    st.markdown("""
    **Phân Tích Chi Tiết Nguyên Nhân**
    
    Để có thể trực quan hơn thì ta chuyển các gam màu trên ma trận HOG của cả bốn frame thành các giá trị số như sau :
    """)
    col1, col2, col3 = st.columns([1,10,1])
    with col2:
        st.image('anh6.png', caption='Bốn frame liên tiếp được dùng để đánh giá')

    st.markdown("""
    *1. Suy Giảm Vùng Đặc Trưng Phía Trên*

    - Frame -3 đến -1: Vùng trên duy trì ổn định với giá trị MED (9000-13000)
    - Frame at loss: Suy giảm mạnh xuống LOW (6000-9000)
    - **Tác động**: *Mất khả năng nhận dạng phần phía trên của đối tượng.*

    *2. Biến Động Vùng HOT (Vùng có mức độ gradient cao)*
    
    - Vị trí thay đổi: Các vùng HOT biến động tại từ (5,1)-(5,2) → (4,1)-(4,2) 
    - Cường độ biến động:
        - Frame -3: Max 20292.50
        - Frame -2: Max 23203.23 (⬆️)
        - Frame -1: Max 22399.94 (⬇️)
        - Frame at loss: Max 22681.33 (↕️)
    - **Tác động**: *Không hề ổn định trong việc xác định đặc trưng mạnh.*

    *3. Suy Giảm Góc Trái Trên*

    Vị trí (0,0) cho thấy sự suy giảm nghiêm trọng nhất (vào góc trái trên) trong quá trình mất tracking.:
    - Cường độ biến động :
        - Frame -3: 13630.63 (HIGH)
        - Frame -2: 13576.10 (HIGH)
        - Frame -1: 13590.78 (HIGH)
        - Frame at loss: 9340.71 (MED) ⬇️ -31.4%
    - **Tác động**: *Thay đổi hoàn toàn giá trị của đặc trưng mạnh.*

    *4. Phân Bố Giá Trị Không Đồng Đều*:


    1. **Vùng HOT**:
       - Giảm từ 19.6% xuống 12.5% (↓ 36.2%)
       - Số lượng giảm từ 3 xuống 2 vùng

    2. **Vùng HIGH**:
       - Giảm từ 25.0% xuống 25.0% (ổn định)
       - Số lượng giảm từ 4 xuống 4 vùng

    3. **Vùng MED**:
       - Giảm từ 31.3% xuống 25.0% (↓ 20%)
       - Số lượng giảm từ 5 xuống 4 vùng

    4. **Vùng LOW**:
       - Không đổi ở 25.0%
       - Số lượng không đổi ở 4 vùng

    5. **Vùng COLD**:
       - Tăng từ 12.5% lên 25.0% (↑ 100%)
       - Số lượng tăng từ 2 lên 4 vùng


    **Nhận xét chi tiết**:


    - *Sự gia tăng vùng COLD từ 12.5% lên 25.0%* là dấu hiệu rõ rệt nhất của việc mất tracking
    - *Sự suy giảm vùng HOT từ 19.6% xuống 12.5%* cho thấy mất dần đặc trưng mạnh
    - *Sự phân bố đồng đều ở frame cuối (mỗi mức 25%)* cho thấy sự mất ổn định của đặc trưng
    """)

    # Kết luận
    st.markdown("""
    *Kết Luận*

    Quá trình mất tracking cơ bản không chỉ diễn ra ở 1 hay 2 frame mà nó liên quan đến quá trình 
    thực hiện dài ở nhiều frame ở trước nữa. Do đó, vẫn có thể đưa ra một số kết luận cho quá trình theo dõi đối
    tượng sử dụng KCF như sau :

    - Quá trình mất tracking cơ bản khó phát hiện bằng mắt thường do sự ảnh hưởng 
    của HOG quá lớn dù chỉ với những thay đổi ánh sáng dù là nhỏ nhất.
    - Nhưng vẫn có thể đánh giá dựa trên sự biến động của các gam màu có trong ma trận HOG.
    - Việc đánh giá trên dựa hoàn toàn vào suy nghĩ cá nhân của sinh viên thực hiện.
    
    """)
        

if __name__ == '__main__':
    main()