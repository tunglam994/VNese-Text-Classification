# Phân loại văn bản Tiếng Việt - Vietnamese text classification

## Mục tiêu
- Nghiên cứu về các phương pháp trong xử lý ngôn ngữ tự nhiên nói chung và xử lý ngôn ngữ tiếng Việt nói riêng
- Thử nghiệm và thực hành một vài hướng tiếp cận cơ bản trong NLP

## Hướng dẫn cài đặt và sử dụng


Dưới đây là hướng dẫn để bắt đầu với dự án của chúng tôi:

1. Clone repository này về máy tính của bạn:

- git clone https: [https://github.com/tunglam994/VNese-Text-Classification.git]
 
3. Tiến hành cài đặt các công cụ và thư viện cần thiết.

4. Mở các tệp và tài liệu hướng dẫn để bắt đầu làm việc.

## Tổng quan bài toán
Nhiệm vụ của bài toán là phân loại các đoạn tin điện tử vào 10 chủ đề:  Chính trị xã hội, Khoa học, Kinh doanh, Pháp luật, Sức khỏe, Thế giới, Thể thao, Vi tính, Văn hóa, Đời sống.

## Kết quả
Hai hướng tiếp cận được sử dụng là sử dụng mô hình xác suất Naive Bayes và mô hình LSTM. Cả 2 hướng tiếp cận đều đạt hiệu quả nhận diện khá tương đồng, điểm F1-score lần lượt là 0.89 (Naive Bayes) và 0.88(LSTM)
