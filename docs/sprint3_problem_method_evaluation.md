# Sprint 3 — Problem, method và evaluation

Tài liệu chọn sáu vấn đề quan trọng nhất của Sprint 3, gồm tổng cộng 12 câu hỏi con
để thuận tiện trình bày. Các số liệu chi tiết hơn được lưu trong `sprint3_report.md`.

## 1. Vấn đề: Thiết lập benchmark GAT/TGAT công bằng và đáng tin cậy

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **1.1. Làm sao so sánh GCN, GraphSAGE, GAT và TGAT trong cùng điều kiện?** | Giữ cố định dataset fingerprint, node split, feature `zero_indicator` 34D, graph direction undirected, seed 42/43/44, batch 1.024, Adam, LR 0,001, weight decay 5e-4, dropout 0,5, tối đa 50 epoch, patience 8, loss có cùng `pos_weight` và checkpoint theo validation AP. | Cả bốn model được đánh giá đủ 183.862 validation node và 183.840 test node; mỗi split có 2.326 fraud node. Test không tham gia chọn checkpoint. | Đây là so sánh công bằng về **protocol** và phù hợp để chọn pipeline tốt nhất. Không yêu cầu số tham số bằng nhau vì mỗi kiến trúc có phép biến đổi và độ phức tạp nội tại khác nhau. |
| **1.2. Vì sao cùng undirected nhưng static model và TGAT dùng cách dựng graph khác nhau?** | GCN/GraphSAGE/GAT thêm cạnh đảo rồi coalesce theo cặp node, tạo 7.994.520 structural edge. TGAT mirror từng temporal event, giữ nguyên timestamp và edge type, tạo 8.601.998 event edge. | Cả hai cách đều cho phép truyền message hai chiều; riêng TGAT không làm mất các event đối ứng xảy ra ở thời điểm khác nhau. | Graph direction được giữ nhất quán, nhưng representation của edge phù hợp với từng pipeline. Vì vậy benchmark phản ánh hiệu quả end-to-end của mỗi pipeline. |

## 2. Vấn đề: Đánh giá hiệu quả của attention tĩnh trong GAT

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **2.1. Cấu hình GAT nào được dùng trong benchmark cuối?** | Dùng GAT hai lớp, tổng hidden width 64, bốn head tương ứng 16 chiều/head, ReLU, hidden dropout 0,5, attention dropout 0 và fan-out `[15,10]`. | Model có 2.435 tham số, gần GCN 2.305 tham số; ba seed đạt validation AP `0,030855 ± 0,000380`. | Cấu hình giữ hidden representation 64 chiều như GCN, GraphSAGE và TGAT, đồng thời tránh áp dụng thêm dropout trực tiếp lên attention coefficient. |
| **2.2. GAT có cải thiện so với GCN trong cùng protocol undirected không?** | So sánh mean của ba seed GAT với GCN undirected, cùng feature, split, sampling depth và training protocol. | GAT đạt test ROC-AUC/AP `0,723304/0,033259`, thấp hơn GCN `0,749892/0,035248` lần lượt `0,026588/0,001989`. | Attention tĩnh chưa tạo lợi thế so với graph convolution trong thiết lập hiện tại. Không nên mặc định kiến trúc phức tạp hơn sẽ cho metric tốt hơn. |

## 3. Vấn đề: Đưa thông tin thời gian vào message passing

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **3.1. TGAT biểu diễn thời gian của cạnh như thế nào?** | Tính `node_time(u)` bằng timestamp nhỏ nhất trong các out-edge gốc của node `u`. Với mỗi sampled edge, dùng `node_time(source) - edge_timestamp`, mã hóa thành vector cosine 16 chiều rồi đưa vào một `TransformerConv` bốn head. | Timestamp tham gia trực tiếp vào attention dưới dạng edge embedding; loader dùng một hop `[15]` phù hợp với một message-passing layer. | TGAT khai thác thứ tự thời gian tương đối của event trong full-history graph, thay vì chỉ xem mọi cạnh là structural edge đồng nhất. |
| **3.2. TGAT hiện tại có phải mô hình dự đoán theo thời gian thực không?** | Đối chiếu sampler và cách tạo input thời gian: dùng static `NeighborLoader`, không có query time và không lọc neighbor theo temporal cutoff. | Model sử dụng timestamp để tạo attention feature nhưng vẫn nhìn graph lịch sử đã quan sát khi phân loại node. | Đây là **full-history transductive node classification**, chưa phải dự đoán fraud tại một thời điểm tương lai và không được diễn giải như temporal forecasting. |

## 4. Vấn đề: Xác định model có chất lượng dự đoán tốt nhất

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **4.1. Model nào đứng đầu theo tiêu chí lựa chọn đã khóa?** | Xếp hạng bằng mean validation AP của ba seed; chỉ đọc test metric sau khi checkpoint đã được chọn. | Validation AP lần lượt là GCN `0,035593`, GraphSAGE `0,038611`, GAT `0,030855` và TGAT `0,040787`. TGAT đứng đầu, hơn GraphSAGE `0,002176`. | TGAT là model được chọn theo đúng validation protocol; test không được dùng để quyết định model. |
| **4.2. Thứ hạng trên test có nhất quán giữa ROC-AUC và AP không?** | Báo cáo cả ROC-AUC và AP do fraud là lớp rất hiếm; tổng hợp mean ± population standard deviation trên seed 42/43/44. | Thứ hạng trên cả hai metric là **TGAT > GraphSAGE > GCN > GAT**. TGAT đạt test `0,784920/0,043885`; GraphSAGE `0,777018/0,043408`; GCN `0,749892/0,035248`; GAT `0,723304/0,033259`. | Kết quả nhất quán giữa hai metric. TGAT dẫn đầu rõ hơn về ROC-AUC; lợi thế AP so với GraphSAGE chỉ là `0,000477`. |

## 5. Vấn đề: Cân bằng chất lượng dự đoán và chi phí mô hình

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **5.1. Số tham số khác nhau có làm benchmark mất công bằng không?** | Giữ cùng training protocol nhưng để mỗi kiến trúc dùng cấu trúc tự nhiên với hidden width 64; báo cáo số tham số như một trade-off thay vì ép parameter matching. | GCN/GAT/GraphSAGE/TGAT lần lượt có 2.305/2.435/4.545/20.001 tham số. Dù GAT gần GCN về capacity, GAT vẫn thấp hơn; model nhiều tham số hơn không tự động bảo đảm metric cao hơn. | Benchmark vẫn hợp lệ để so sánh pipeline end-to-end. Tuy nhiên, không thể quy toàn bộ chênh lệch cho riêng attention hoặc time encoding vì capacity và phép toán nội tại cũng khác nhau. |
| **5.2. TGAT hay GraphSAGE là lựa chọn thực tế hơn?** | So sánh test metric, tổng runtime ba seed và số tham số của hai model đứng đầu. | TGAT hơn GraphSAGE `0,007902` test ROC-AUC và `0,000477` test AP, nhưng có 20.001 so với 4.545 tham số và runtime 3.910,6 so với 2.710,9 giây, tức chậm hơn khoảng 1,44 lần. | TGAT phù hợp khi ưu tiên metric cao nhất; GraphSAGE là lựa chọn cân bằng hơn khi ưu tiên mô hình nhẹ, nhanh và test AP gần tương đương. |

## 6. Vấn đề: Bảo đảm kết luận đúng phạm vi

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **6.1. Kết quả có ổn định và tránh test leakage không?** | Chạy đủ ba seed, đánh giá toàn bộ validation/test split, chọn checkpoint duy nhất bằng validation AP và báo mean ± population standard deviation. | TGAT có validation ROC-AUC/AP std `0,000564/0,000145` và test ROC-AUC/AP std `0,001017/0,000303`; thứ hạng tổng thể không phụ thuộc vào một seed đơn lẻ. | Kết quả có độ ổn định tốt trong ba seed và test không tham gia tuning. Ba seed vẫn là bằng chứng thực nghiệm, không thay thế kiểm định thống kê quy mô lớn. |
| **6.2. Sprint 3 đã chứng minh phát hiện fraud ring hoặc fraud tương lai chưa?** | Đối chiếu task, output và phạm vi dữ liệu mà model thật sự sử dụng. | Các model trả xác suất fraud ở cấp node. TGAT dùng timestamp trong attention nhưng không có temporal cutoff; pipeline chưa tạo community/ring hoặc đánh giá dự báo tương lai. | Chưa thể kết luận đã phát hiện fraud ring hay dự báo fraud tương lai. Kết quả chỉ chứng minh hiệu quả node classification full-history trên DGraphFin. |

## Kết luận ngắn

- Benchmark giữ công bằng về protocol; số tham số được xem là trade-off tự nhiên của
  từng kiến trúc, không phải điều kiện bắt buộc phải bằng nhau.
- GAT đạt test ROC-AUC/AP `0,723304/0,033259` và chưa vượt GCN trong cùng thiết lập
  undirected.
- TGAT đứng đầu trên cả validation AP và hai test metric; thứ hạng cuối là **TGAT >
  GraphSAGE > GCN > GAT**.
- Lợi thế AP của TGAT so với GraphSAGE nhỏ (`+0,000477`) trong khi TGAT lớn và chậm
  hơn; lựa chọn cuối phụ thuộc ưu tiên chất lượng hay chi phí.
- Kết quả thuộc full-history transductive node classification, chưa phải temporal
  forecasting hoặc fraud-ring extraction.
