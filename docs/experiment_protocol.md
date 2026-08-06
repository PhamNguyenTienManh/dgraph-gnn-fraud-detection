# Protocol thực nghiệm Sprint 2

Protocol này đã được hiện thực hóa cho GCN, GraphSAGE và RGCN. Các cấu hình full raw 17 chiều, zero-indicator 34 chiều và RGCN background-aware 34 chiều đã chạy đủ ba seed.

## Thiết lập bài toán

- Bài toán: phân loại node nhị phân, normal (0) và fraud (1).
- Positive class: 1.
- Lớp 2/3 được giữ trong graph cho message passing, không dùng trong loss/metric.
- Split: dùng nguyên các mảng chỉ số 70/15/15 được cung cấp.
- Thiết lập baseline: static, transductive; timestamp được bảo toàn nhưng chưa đưa vào message passing.
- Dữ liệu canonical giữ cạnh có hướng `source -> target`.

## Metric và lựa chọn mô hình

- Metric lựa chọn checkpoint: validation Average Precision (AP), giá trị lớn hơn tốt hơn.
- Metric bắt buộc báo cáo: validation/test AP và ROC-AUC.
- Test chỉ chạy sau khi khóa cấu hình và chọn checkpoint bằng validation.
- Mọi class weight phải tính duy nhất từ train; không dùng thống kê nhãn validation/test để fit.

## Cấu hình khởi đầu theo tài nguyên

| Tham số | Giá trị khởi đầu |
|---|---|
| Device | CPU |
| Sampler | PyG `NeighborLoader` dùng backend `pyg-lib`; CSR sampler là fallback |
| Fan-out | `[15, 10]` |
| Batch size | 1.024 seed node |
| Worker | 0 trên Windows |
| Seed chính | 42 |
| Số seed kết quả | 42, 43, 44 |
| Feature | Raw anonymized values, đổi sang `float32`; không normalize mặc định |

## Biến thể được phép đánh giá bằng validation

1. Raw feature so với xử lý missing/normalization có tài liệu rõ ràng.
2. Loss không trọng số so với positive class weight tính trên train.
3. Cạnh theo hướng gốc so với cạnh đảo/hai chiều; hướng gốc luôn là canonical baseline.
4. Batch size nhỏ hơn nếu RAM tăng cao khi có activation và optimizer state.

## Điều kiện so sánh GCN, GraphSAGE và RGCN

- Cùng fingerprint dataset, split, feature policy, sampler seed và target batches.
- Cùng optimizer, learning rate, số epoch tối đa, patience và quy tắc checkpoint trong thí nghiệm so sánh chính.
- Hidden dimension, số layer và dropout ưu tiên giống nhau; ngoại lệ phải được ghi trong run manifest.
- So sánh GCN/RGCN ưu tiên khớp số tham số: GCN hidden 64 có 2.305 tham số; RGCN hidden 13 có 2.289 tham số vì mỗi relation có ma trận riêng.
- Báo cáo thời gian/epoch, peak RAM quan sát được và tổng thời gian huấn luyện bên cạnh metric.

## Quyết định đã áp dụng

- Backend đã chốt: PyTorch 2.12 CPU + PyG 2.8.0.post1 + `pyg-lib` 0.8.0, dùng `NeighborLoader` trên Windows.
- Baseline giữ cạnh theo hướng gốc; `NeighborLoader` lấy láng giềng phù hợp với luồng message passing của PyG. Đảo cạnh/hai chiều chưa được đánh giá.
- Hai cấu hình full dùng cùng ba seed 42, 43, 44 và cùng ngân sách huấn luyện/đánh giá.
- Baseline chính giữ raw feature. Biến thể `zero_indicator` thay `-1` bằng 0 và nối thêm 17 cờ missing, tạo 34 chiều; hai biến thể được so sánh bằng validation trước khi báo cáo test.
- RGCN paper-style ghi đè 11 edge type gốc bằng bốn relation `T→T`, `T→B`, `B→T`, `B→B`; T gộp normal/fraud nên relation không tiết lộ fraud label. Cấu hình này đo lợi ích của background-aware message passing, không phải ý nghĩa 11 loại quan hệ khẩn cấp.
- `pos_weight = N_normal / N_fraud` được tính chỉ từ train và AP validation quyết định checkpoint.
