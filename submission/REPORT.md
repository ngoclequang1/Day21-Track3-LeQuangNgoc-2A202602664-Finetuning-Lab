# Lab 21 — Evaluation Report

**Họ tên**: Lê Quang Ngọc  
**MSSV**: 2A202602664  
**Ngày**: 07/10/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: NVIDIA Tesla T4, 14,6 GB khả dụng, huấn luyện bằng fp16

## 1. Lựa chọn và thiết lập

Tôi dùng bộ dữ liệu mặc định gồm 250 ticket chăm sóc khách hàng tiếng Việt, với đầu ra JSON có bốn trường `intent`, `urgency`, `product` và `sentiment`. Tôi chọn bộ dữ liệu này vì nhãn có thể được chấm khách quan theo từng trường, định dạng JSON có thể kiểm tra tự động, và pipeline đã có thêm tập regression để đo sự suy giảm năng lực chung. Dữ liệu được chia thành 225 mẫu train và 25 mẫu validation với seed 42.

Tôi chọn Qwen3.5-4B vì đây là model mặc định phù hợp với Tesla T4 của Colab. T4 không hỗ trợ bf16, nên pipeline tự dùng fp16 với gradient scaling. Tôi huấn luyện 2 epoch, tương ứng 30 optimizer step, với `MASK_MODE=assistant-only`.

| Thiết lập | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / validation | 225 / 25, seed 42 |
| Độ dài token p95 | 98 |
| `max_length` | 1024 |
| `MASK_MODE` | `assistant-only` |
| Epoch / optimizer steps | 2 / 30 |

`token_stats.json` gợi ý `max_length=256`, nhưng tôi giữ 1024 theo cấu hình chuẩn của tier T4 để mọi run dùng cùng điều kiện và vẫn hỗ trợ đầu vào dài hơn. Đổi xuống 256 có thể tiết kiệm activation memory, nhưng việc thay đổi cấu hình sau khi đã đóng băng baseline sẽ làm phép so sánh khó diễn giải hơn.

Template giữ được khối reasoning: `template_check.json` có `ok=true`, `open_tag_present=true`, `body_present=true` và verdict `reasoning preserved — safe to train on traces`. Tuy vậy, corpus được cung cấp chỉ chứa câu trả lời JSON, nên `valid_trace_rate=0.0` không được dùng để kết luận model mất năng lực suy luận.

## 2. Bằng chứng loss mask

| Kiểm tra | Kết quả |
|---|---:|
| `supervised_fraction` | 0.4149 |
| Số token supervised / tổng | 39 / 94 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Kết quả này chứng minh loss tập trung vào phần assistant và stop token, thay vì bắt model học lại system prompt và câu hỏi. `supervised_fraction` thấp hơn 0.95 nên không rơi vào lỗi tính loss trên toàn bộ hội thoại.

## 3. Mốc đánh giá đóng băng và ba baseline

NB2 được chạy trước huấn luyện trên đủ 50 mẫu target và 15 mẫu regression. Prompt tối ưu giữ nguyên checksum `719e74d3b6232053`; tôi không sửa hoặc làm yếu baseline này.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.000 | 0.7911 | 0.000 | 3236.7 |
| (b) Base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1010.1 |
| (c) LoRA fine-tune | **0.965** | 0.7444 | 1.000 | 1415.6 |

Baseline (b) thực sự mạnh hơn (a): target tăng từ 0.000 lên 0.765, format tăng từ 0 lên 1.000 và latency giảm mạnh. Vì vậy, fine-tune được so với một prompt mạnh và hợp lệ, không phải một baseline cố tình yếu. Fine-tune tiếp tục tăng target thêm 0.200 so với (b), giữ format hoàn hảo, nhưng chậm hơn prompt tối ưu khoảng 405,5 ms mỗi mẫu và làm regression giảm 0.0467.

## 4. Giải phẫu cấu hình

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Thời gian (s) | VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear, 12 module | 16 | 32.464.896 | 1e-4 | 0.6293 | 0.965 | 402.2 | 8.78 |
| `attn_only` | q,v, 2 module | 283 | 32.456.704 | 1e-4 | **0.5379** | **0.970** | 268.9 | 8.79 |
| `wrong_lr` | text-linear, 12 module | 16 | 32.464.896 | 1e-5 | 1.5702 | 0.000 | 399.0 | 8.78 |
| `qlora` | text-linear, 12 module, 4-bit | 16 | 32.464.896 | 1e-4 | 0.7058 | 0.940 | 468.3 | **3.86** |

### 4.1 Vị trí adapter so với rank

`attn_only` đạt target 0.970, nhỉnh hơn `correct` 0.965 đúng 0.005. Hai run có ngân sách gần như bằng nhau: 32.456.704 so với 32.464.896 tham số, sai lệch chỉ khoảng 0,025%, nên đây là đối chứng công bằng. Thứ tự theo target giống thứ tự theo train loss trong lần chạy này vì `attn_only` cũng có loss thấp hơn, 0.5379 so với 0.6293. Tuy nhiên, chênh lệch target chỉ bằng nửa điểm phần trăm nên kết quả không chứng minh attention-only luôn tốt hơn; nó cho thấy trên tác vụ triage hẹp này, tăng rank để khớp ngân sách có thể bù cho phạm vi gắn adapter hẹp. Vì vậy, tôi không thể khẳng định vị trí luôn thắng rank chỉ từ phép đo này; kết luận đúng là hai cấu hình gần như hòa về chất lượng target, trong khi `attn_only` nhanh hơn khi train và suy luận.

### 4.2 Learning rate sai

`wrong_lr` chỉ giảm learning rate từ 1e-4 xuống 1e-5 nhưng train loss dừng ở 1.5702, cao hơn nhiều so với 0.6293 của `correct`. Quan trọng hơn, target và format đều bằng 0, đồng thời latency tăng lên 5371,8 ms/mẫu vì model không học được cách phát JSON ngắn và hợp lệ. Nếu chỉ thấy loss đang đi xuống mà không biết LR và không chấm target, tôi có thể kết luận sai rằng training vẫn đang hội tụ và chỉ cần thêm step. Phép đo cho thấy LR ở thang full fine-tuning là quá nhỏ cho LoRA trong ngân sách 30 step; đây là nút vặn có tác động lớn nhất trong bốn cấu hình.

### 4.3 QLoRA

QLoRA giảm peak VRAM từ 8,78 GB xuống 3,86 GB, tiết kiệm 4,92 GB, tương đương khoảng 56,0%. Đổi lại, target giảm từ 0.965 xuống 0.940, loss tăng từ 0.6293 lên 0.7058, latency tăng từ 1415,6 lên 1791,5 ms/mẫu và thời gian train tăng từ 402,2 lên 468,3 giây. Format vẫn đạt 1.000, nên lượng tử hóa không phá cấu trúc đầu ra nhưng tạo ra một chi phí chất lượng nhỏ và đo được. Trên T4, LoRA fp16 chỉ dùng 8,78/14,6 GB nên QLoRA không cần thiết cho model 4B này. Số đo của tôi vì thế ủng hộ khuyến nghị không dùng QLoRA cho Qwen3.5-4B khi LoRA 16-bit vẫn vừa GPU; QLoRA chỉ đáng cân nhắc nếu giới hạn VRAM buộc phải đánh đổi.

## 5. Phán quyết

**Kết quả cổng hồi quy: FAILED**  
`target Δ = +0.200` · `regression Δ = -0.0467` · `valid_trace_rate = 0.0`

Fine-tune học tác vụ mục tiêu rất tốt: target tăng từ 0.765 lên 0.965 và format vẫn đạt 1.000. Tuy nhiên, cổng yêu cầu regression không giảm quá 0.020, trong khi kết quả giảm 0.0467. Vì vậy, verdict FAILED là đúng dù độ chính xác triage tăng mạnh. Đây không phải thất bại của pipeline: mask đúng, baseline đã đóng băng, prompt tối ưu không bị sửa và bốn run có cùng ngân sách step. Nó là bằng chứng rằng 225 mẫu chuyên biệt có thể đẩy hành vi của model quá mạnh về một tác vụ và làm giảm năng lực chung. Tôi không nới tolerance hay thay tập eval để đổi verdict. Bước tiếp theo hợp lý là thêm 1–5% replay data phổ thông vào train, đóng băng một mốc mới trước khi huấn luyện lại, rồi kiểm tra liệu target có được giữ trong khi regression phục hồi hay không.

## 6. Phân tích định tính

`qualitative.json` lưu output của fine-tune nhưng không lưu prediction từng mẫu của baseline (b); do đó tôi không dựng lại hoặc bịa nội dung baseline. Cột (b) dưới đây ghi rõ trạng thái artefact, còn kết luận thắng tổng thể dựa trên target aggregate 0.765 so với 0.965. Hai ca thua được xác định trực tiếp từ `ft_score=0.75`.

| # | Ticket rút gọn | Nhãn đúng | (b) optimized prompt | (c) Fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | Chuột không dây, trả lại, gấp, shop hỗ trợ tốt | `doi_tra / cao / chuột không dây / tich_cuc` | Prediction không được lưu; aggregate target 0.765 | Khớp cả 4 trường (`ft_score=1.00`) | ✅ FT đúng hoàn toàn |
| 2 | Ốp lưng điện thoại, hoàn tiền, sớm, bực mình | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | Prediction không được lưu; aggregate target 0.765 | Khớp cả 4 trường (`ft_score=1.00`) | ✅ FT đúng hoàn toàn |
| 3 | Đèn bàn LED, hoàn tiền, quá hạn, cảm ơn shop | `hoan_tien / cao / đèn bàn LED / tich_cuc` | Prediction không được lưu; aggregate target 0.765 | Khớp cả 4 trường (`ft_score=1.00`) | ✅ FT đúng hoàn toàn |
| 4 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện, cảm ơn | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Prediction không được lưu; aggregate target 0.765 | Dự đoán urgency `trung_binh`, đúng 3/4 trường | ❌ FT thua ở urgency |
| 5 | Nồi chiên không dầu, thiếu phụ kiện, khi nào tiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | Prediction không được lưu; aggregate target 0.765 | Dự đoán urgency `trung_binh`, đúng 3/4 trường | ❌ FT thua ở urgency |

Cả hai ca fine-tune thua đều sai trường `urgency`: cụm “khi nào tiện” phải ánh xạ thành `thap`, nhưng model dự đoán `trung_binh`. `intent`, `product` và `sentiment` vẫn đúng. Mẫu chung này cho thấy phần còn yếu không phải JSON hay nhận diện sản phẩm, mà là ranh giới ngữ nghĩa giữa mức khẩn cấp thấp và trung bình. Tôi sẽ ưu tiên bổ sung các cặp tối thiểu chỉ khác dấu hiệu urgency, thay vì tăng rank một cách tổng quát.

## 7. Kết luận và điều học được

Tôi chưa nên deploy adapter này ở trạng thái hiện tại. Về tác vụ triage, kết quả rất mạnh: target đạt 0.965, cao hơn prompt tối ưu 0.200, và mọi output đều đúng định dạng JSON. Tuy nhiên, regression giảm 0.0467, hơn hai lần ngưỡng cho phép 0.020. Một hệ thống production không chỉ cần làm tốt luồng chính mà còn phải tránh biến mọi yêu cầu ngoài miền thành hành vi triage hoặc làm suy giảm khả năng trả lời chung. Kết quả đối chứng cũng cho thấy learning rate là đòn bẩy rõ nhất: giảm LR mười lần làm target và format cùng về 0. QLoRA tiết kiệm nhiều VRAM nhưng không cần thiết trên T4 vì LoRA fp16 vẫn vừa bộ nhớ và cho chất lượng tốt hơn. `attn_only` nhỉnh hơn `correct` 0.005 ở ngân sách tham số khớp, nên trong thí nghiệm này tôi không có bằng chứng rằng mở rộng vị trí adapter quan trọng hơn rank; chênh lệch quá nhỏ để khái quát. Bước cải thiện có giá trị nhất không phải tăng rank mà là sửa phân phối dữ liệu: thêm 1–5% replay data để chống quên, đồng thời tăng ví dụ phân biệt urgency thấp và trung bình. Sau đó tôi sẽ chạy lại toàn bộ baseline và regression gate trước khi cân nhắc deploy.

Ba điều tôi học được:

1. Target tăng mạnh không đồng nghĩa model an toàn để triển khai; regression gate đã bác một adapter đạt target 0.965 vì năng lực chung giảm quá mức.
2. Learning rate của LoRA có tác động lớn hơn việc chuyển từ text-linear sang attention-only trong ngân sách thí nghiệm này: LR sai làm target về 0, còn hai vị trí adapter chỉ lệch 0.005.
3. Loss mask phải được chứng minh ở mức token. Ở đây chỉ 41,49% token được supervise, câu hỏi được loại khỏi loss và câu trả lời cùng stop token vẫn nằm trong loss.

Nếu có thêm hai giờ, tôi sẽ thêm 1–5% replay data phổ thông, giữ nguyên model và cấu hình LoRA, rồi chạy lại với cùng eval checksum. Tôi cũng sẽ bổ sung các mẫu đối chứng cho urgency và lưu prediction theo từng mẫu của baseline (b) để phân tích định tính trực tiếp hơn.

## Phụ lục — phần thưởng

- [ ] B1 — merge và hot-swap adapter
- [ ] B2 — dataset miền riêng
- [ ] B3 — reasoning-trace collapse
- [ ] B4 — quét rank có kiểm soát
- [ ] B5 — Hugging Face Hub
