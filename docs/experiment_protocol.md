# Protocol thực nghiệm GNN

## Thiết lập bài toán

- Dataset: DGraphFin, xác minh bằng SHA-256.
- Task: binary node classification; nhãn 0 normal, 1 fraud; node nền không tham gia
  loss/metric.
- Split: giữ nguyên train/validation/test index từ dataset.
- Feature chính: `zero_indicator` 34 chiều, không standardize.
- Loss: `BCEWithLogitsLoss`; `pos_weight` chỉ tính từ train.
- Checkpoint: chọn duy nhất bằng validation Average Precision.
- Test: chỉ đánh giá sau khi checkpoint đã được chọn.
- Full run: seed 42, 43, 44; báo mean ± population standard deviation.

## Training budget chung

| Thành phần | Giá trị |
|---|---|
| Batch size | 1.024 seed node |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Weight decay | 5e-4 |
| Dropout | 0,5 |
| Epoch tối đa | 50 |
| Early-stopping patience | 8 |
| Gradient clipping | 2,0 |
| Worker | 0 trên Windows |

Loss và metric chỉ tính trên seed node đầu batch; sampled neighbor chỉ cung cấp
context cho message passing. Full evaluation đi qua toàn bộ validation/test split.

## GCN, GraphSAGE và GAT undirected

- Graph dùng `structural_coalesced`: thêm cạnh đảo và coalesce theo cặp node.
- Fan-out `[15,10]` khớp hai message-passing layer.
- GAT dùng bốn head với tổng hidden width 64, ReLU, hidden dropout 0,5 và attention
  dropout 0; tổng cộng 2.435 tham số.
- GCN/GraphSAGE dùng kết quả undirected đã chạy ở Sprint 2; GAT dùng full run Sprint 3.
- Cùng feature, split, seed, batch và training budget ở trên.

## TGAT undirected

- Tên model chính thức: `tgat`.
- Graph dùng `temporal_event_mirror`: mỗi event `u→v` tại thời điểm `t` tạo thêm
  event `v→u` với cùng timestamp và edge type. Event đối ứng ở thời điểm khác không
  bị coalesce.
- `node_time(u)` được tính trên graph có hướng gốc, trước khi mirror, bằng timestamp
  nhỏ nhất trong các out-edge gốc của `u`; node không có out-edge nhận 0.
- Mỗi sampled edge dùng `node_time(source) - edge_timestamp` làm time input.
- Cosine `TimeEncode` tạo edge embedding cho một PyG `TransformerConv`.
- Model có một message-passing layer nên sampler dùng đúng một hop `[15]`.
- Loader là static `NeighborLoader`; timestamp tham gia attention nhưng không lọc
  edge. Không có query time hoặc recursive temporal cutoff.

## Phạm vi so sánh công bằng

Bảng chính chỉ so GCN, GraphSAGE, GAT và TGAT **undirected**. Các yếu tố dataset,
split, feature, seed, batch, optimizer, training budget, loss và checkpoint selection
được khóa. Tuy nhiên, static model dùng structural coalescing và hai hop `[15,10]`,
còn TGAT bảo toàn temporal event và dùng một hop `[15]`. Vì thế chỉ diễn giải kết quả
là hiệu quả end-to-end của từng pipeline trong cùng graph direction; không quy chênh
lệch cho riêng timestamp, attention hay sampler.

## Provenance và nghiệm thu

- Mỗi run lưu config, checkpoint, metric theo epoch, environment, parameter count,
  runtime, dataset hash và trạng thái full/partial.
- Full result phải đủ ba seed, đủ validation/test node và không dùng test để tuning.
- Notebook không chứa error output; cờ official mặc định là `False`.
- Catalog Sprint 3 chỉ giữ `gat_undirected` và `tgat_undirected`.
- TGAT strict không thuộc protocol hiện tại. TGAT `[15,10]` cũ không thuộc nội dung
  chính và raw artifact đã được xóa.

Kết quả là full-history transductive node classification. Không diễn giải thành dự
đoán fraud tương lai khi dataset không có timestamp xác lập nhãn, hoặc thành phát hiện
nhóm gian lận có phối hợp.
