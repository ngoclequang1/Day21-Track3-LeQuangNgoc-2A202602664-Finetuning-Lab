# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là fine-tune tăng target từ 0.765 lên 0.965 và giữ format hoàn hảo, nhưng vẫn bị cổng đánh giá kết luận FAILED. Regression giảm 0.0467 cho thấy một kết quả trông rất tốt trên tác vụ chính vẫn có thể chưa đủ an toàn để triển khai.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở NB4 vì phải huấn luyện ba cấu hình đối chứng trên cùng ngân sách 30 step. Tôi đã dự đoán training sẽ là phần lâu nhất, nhưng không ngờ phần đánh giá nhiều lượt sinh văn bản và việc tải, khôi phục artefact trên Colab cũng cần nhiều thời gian như vậy.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab tôi có xu hướng tin rằng target accuracy tăng rõ rệt là đủ để xem fine-tune thành công. Sau thí nghiệm này, tôi không còn tin một chỉ số target hoặc train loss riêng lẻ có thể quyết định việc deploy; regression, format và latency đều có thể làm thay đổi kết luận.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc yêu cầu, chuẩn bị notebook Colab, giải thích log fp16 của T4, hướng dẫn tải artefact và tổng hợp report từ các file JSON. Điểm cần thận trọng là AI không thể tự suy ra prediction từng mẫu của baseline (b) khi pipeline không lưu chúng; nếu tự điền nội dung đó thì sẽ thành bịa dữ liệu. Vì vậy report ghi rõ giới hạn của artefact thay vì giả định output.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ xác định trước tập target, tập regression và tiêu chí deploy, sau đó đóng băng chúng trước khi huấn luyện. Đồng thời tôi sẽ kiểm tra chat template và giải mã ngược loss mask trên vài mẫu thật, vì nếu dữ liệu hoặc mask sai thì mọi phép tối ưu sau đó đều không còn ý nghĩa.
