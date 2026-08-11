# Sprint 2 — Problem, method và evaluation

Tài liệu chọn sáu vấn đề quan trọng nhất của Sprint 2, gồm tổng cộng 11 câu hỏi con để thuận tiện trình bày. Các số liệu chi tiết hơn được lưu trong `sprint2_report.md`.

## 1. Vấn đề: Huấn luyện graph lớn đúng cách trong giới hạn tài nguyên

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **1.1. Neighbor sampling có thể train và đánh giá đủ dữ liệu thật hay không?** | Dùng PyG `NeighborLoader`/`pyg-lib`, batch 1.024, fan-out `[15, 10]`, hai hop tương ứng hai lớp GNN và `num_workers=0`. Train lấy `input_nodes` từ train split; validation/test dùng đúng split tương ứng. | Mỗi epoch đi qua đủ 857.899 train seed node; validation và test đánh giá đủ 183.862/183.840 seed node. Các run GCN, GraphSAGE và RGCN đều hoàn tất. | Đây là sampled mini-batch training, không phải full-batch. Loss/metric chỉ tính trên `batch.batch_size` seed node; neighbor, kể cả node lớp 2/3, chỉ tham gia message passing. |
| **1.2. Làm sao xử lý mất cân bằng fraud mà không làm sai sampling?** | Giữ nguyên phân bố seed và dùng `BCEWithLogitsLoss` với `pos_weight = N_normal / N_fraud`, chỉ tính từ train. Báo cáo ROC-AUC/AP và chọn checkpoint bằng validation AP. | Train có 847.042 normal và 10.857 fraud nên `pos_weight = 78,018`. Toàn epoch vẫn đi qua mỗi train seed một lần. | `pos_weight` là loss weighting, không phải oversampling; từng batch/subgraph không bị ép giữ tỷ lệ fraud. AP phù hợp hơn accuracy khi lớp fraud rất hiếm. |

## 2. Vấn đề: Chọn kiến trúc và cách biểu diễn missing value

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **2.1. GCN hay GraphSAGE phù hợp hơn với baseline DGraphFin?** | So sánh GCN và GraphSAGE raw 17D trên cùng split, seed 42/43/44, hidden size 64, hai lớp, dropout 0,5, fan-out, optimizer và ngân sách. | GraphSAGE đạt test `0,758562/0,038309` ROC-AUC/AP, cao hơn GCN `0,681957/0,027691`; cả ba paired seed đều cùng xu hướng. | GraphSAGE là baseline tốt hơn trong protocol hiện tại, nhưng đây vẫn là node classification static/transductive. |
| **2.2. Thêm cờ đánh dấu missing value có quan trọng không?** | So sánh raw 17D với zero-indicator 34D: thay `-1` bằng 0 rồi nối thêm 17 cờ missing; giữ nguyên các yếu tố thực nghiệm khác. | Với GCN, 34D tăng test ROC-AUC/AP `+0,003673/+0,000750`. Với GraphSAGE, test AP tăng `+0,000292`, còn ROC-AUC thay đổi `-0,000010` gần như hòa; validation AP tăng ở cả hai model. | Missing indicator có ích nhưng mức tăng nhỏ, rõ hơn với GCN. Không nên kết luận tuyệt đối vì mới có ba seed và số tham số lớp đầu gần gấp đôi. |

## 3. Vấn đề: Xác định vai trò của background node

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **3.1. Background node có thực sự quan trọng trong cấu trúc graph không?** | Phân loại cạnh theo loại node ở hai endpoint: `T→T`, `T→B`, `B→T`, `B→B`; đếm số cạnh của từng relation. | 82,649% cạnh có ít nhất một background endpoint; chỉ 17,351% cạnh là `T→T`. | Background không có supervised label 0/1 nhưng chiếm phần lớn cấu trúc kết nối và có thể cung cấp ngữ cảnh qua message passing. |
| **3.2. Mã hóa rõ vai trò target/background có cải thiện mô hình không?** | So sánh GCN 34D với RGCN-BG 34D dùng bốn relation endpoint; parameter-match RGCN hidden 13 (2.289 tham số) với GCN hidden 64 (2.305 tham số). | RGCN-BG tăng test ROC-AUC/AP `+0,080905/+0,012730`; cao hơn GCN trên cả bốn metric và cả ba paired seed. | Kết quả chứng minh **mã hóa vai trò background có ích**, chưa chứng minh giữ background tốt hơn xóa. Muốn kết luận giữ/xóa cần ablation loại bỏ background riêng. |

## 4. Vấn đề: Xác định yếu tố nào thật sự cải thiện GCN

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **4.1. Trong các ablation đã thử, thay đổi nào cải thiện GCN rõ nhất?** | Thực hiện one-factor-at-a-time trên GCN zero-indicator 34D: lần lượt thử z-score, dropout `0,5 → 0`, weight decay `5e-4 → 5e-7`, learning rate `0,001 → 0,01` và graph undirected. Với undirected, thêm cạnh ngược rồi coalesce để message passing theo cả hai chiều. | Z-score không có lợi; dropout 0 tạo trade-off nhỏ; weight decay thấp chỉ cải thiện nhẹ; learning rate `0,01` làm metric giảm. Undirected tăng test ROC-AUC/AP từ `0,685631/0,028441` lên `0,749892/0,035248`, là mức tăng lớn và nhất quán nhất. | Undirected có thể cải thiện mạnh vì node nhận được thông tin từ cả hàng xóm nối vào lẫn hàng xóm nối ra, làm ngữ cảnh cục bộ đầy đủ hơn và giảm mất thông tin do chỉ truyền theo một hướng. Đây là lời giải thích hợp lý từ cơ chế message passing và mức tăng số cạnh từ 4.300.999 lên 7.994.520, chưa phải chứng minh quan hệ gốc thực sự vô hướng. Vì vậy cấu hình chỉ chọn undirected và giữ các siêu tham số baseline còn lại. |

## 5. Vấn đề: Chọn cấu hình mạnh nhất mà không diễn giải quá mức

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **5.1. Lợi ích của graph undirected có áp dụng cho GraphSAGE không?** | Train lại GraphSAGE 34D trên graph undirected với cùng protocol và ba seed, rồi so sánh paired với GraphSAGE directed. | Test ROC-AUC/AP tăng từ `0,758553/0,038601` lên `0,777018/0,043408`; cả ba seed đều tăng. Trong cùng graph undirected, GraphSAGE cao hơn GCN `+0,027126/+0,008161`. | GraphSAGE 34D undirected là cấu hình tốt nhất trong nhánh GCN/GraphSAGE đã thử. |
| **5.2. Có thể xem RGCN-BG là bước tiếp theo của cấu hình undirected không?** | Đối chiếu config và artifact thay vì xếp mọi kết quả thành một chuỗi cải tiến. | RGCN-BG directed đạt test `0,766536/0,041171`; GraphSAGE 34D undirected đạt `0,777018/0,043408`. Hai kết quả đến từ hai nhánh thí nghiệm độc lập. | Không cộng dồn mức tăng của RGCN-BG và undirected. RGCN undirected chưa được chạy nên chưa biết hai yếu tố có bổ trợ nhau hay không. |

## 6. Vấn đề: Bảo đảm kết luận đáng tin cậy và đúng phạm vi

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **6.1. Kết quả có tái kiểm tra được và có tránh test leakage không?** | Dùng ba seed; chọn checkpoint chỉ bằng validation AP; test sau khi khóa checkpoint. Lưu config, SHA-256 dữ liệu, phiên bản môi trường, log epoch, checkpoint, `metrics.json` và `comparison.json`; cố định evaluation seed. | Artifact có cấu trúc thống nhất, aggregate mean/std được lưu. Ba notebook đã chạy lại không có error output; kích thước valid/test được đánh giá đầy đủ. | Test không tham gia lựa chọn. Neighbor sampling vẫn có thể tạo dao động nhỏ; ba seed là bằng chứng ban đầu chứ chưa phải kiểm định thống kê tuyệt đối. |
| **6.2. Kết quả hiện tại đã chứng minh phát hiện fraud ring động chưa?** | Đối chiếu output và input thật sự của pipeline với mục tiêu bài toán fraud-ring detection. | Pipeline dự đoán xác suất fraud ở cấp node; GCN/GraphSAGE dùng graph static; RGCN-BG chỉ dùng bốn relation target/background. Timestamp và 11 edge type gốc chưa tham gia message passing. | Chưa thể kết luận đã phát hiện fraud ring, quan hệ động hoặc ý nghĩa của edge type gốc. Các bước tiếp theo là ring/community extraction, temporal evaluation, relation gốc và background-removal ablation. |

## Kết luận ngắn

- Sampling và loss đã được triển khai đúng vai trò: sampler tạo ngữ cảnh, còn supervised loss chỉ tính trên seed node.
- `pos_weight` xử lý mất cân bằng tại loss mà không thay đổi phân bố sampling.
- Missing indicator có lợi nhỏ; relation target/background và message passing hai chiều tạo mức cải thiện lớn hơn.
- GraphSAGE 34D undirected là cấu hình tốt nhất trong nhánh GCN/GraphSAGE đã thử.
- Kết quả vẫn thuộc node-level static/transductive classification, chưa phải kết luận hoàn chỉnh về fraud-ring detection.
