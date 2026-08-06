# Sprint 1 — Problem, method và evaluation

Tài liệu tập trung vào sáu vấn đề cốt lõi của giai đoạn hiểu dữ liệu và xây dựng graph pipeline, gồm 12 câu hỏi con. Phần neighbor sampling và cách tính loss trên seed node được trình bày trong `sprint2_problem_method_evaluation.md`, nơi các nội dung đó được kiểm chứng bằng quá trình huấn luyện hoàn chỉnh.

## 1. Vấn đề: Chưa có data contract đáng tin cậy cho DGraphFin

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **1.1. Dataset gồm những trường nào và shape/kiểu dữ liệu thực tế ra sao?** | Đọc trực tiếp metadata của file NPZ; đối chiếu `x`, `y`, `edge_index`, `edge_type`, `edge_timestamp` và ba trường split; ghi lại thành data dictionary. | Xác nhận 3.700.550 node, 4.300.999 cạnh, 17 feature, 11 edge type và 821 timestamp. `x` nguồn là `float64`; các trường còn lại là `int64`. | `train_mask`, `valid_mask`, `test_mask` thực tế là mảng node index, không phải boolean mask dù tên trường dễ gây hiểu nhầm. |
| **1.2. Làm sao biết đúng phiên bản dữ liệu được dùng xuyên suốt thí nghiệm?** | Tính SHA-256 và kích thước file nguồn; lưu fingerprint vào báo cáo profiling/artifact. | File có kích thước 680.317.982 byte và SHA-256 `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85`. | Fingerprint giúp phát hiện dữ liệu bị thay đổi và bảo đảm các sprint sau dùng cùng một nguồn dữ liệu. |

## 2. Vấn đề: Ý nghĩa nhãn và tính đúng đắn của split chưa rõ

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **2.1. Bốn nhãn 0/1/2/3 được dùng thế nào trong bài toán?** | Thống kê toàn bộ `y`; đối chiếu mô tả dataset và xác định target class so với background class. | Có 1.210.092 normal (0), 15.509 fraud (1), 1.620.851 background (2) và 854.098 background (3). | Bài toán supervised là phân loại node 0/1. Node 2/3 được giữ trong graph để cung cấp cấu trúc, nhưng không thuộc loss/metric nhị phân. |
| **2.2. Train/validation/test có chồng lấn, sai nhãn hoặc bỏ sót target node không?** | Kiểm tra miền node ID, phần tử trùng trong từng split, giao giữa ba split, nhãn tại các index và độ bao phủ toàn bộ node lớp 0/1. | Ba split không chồng lấn, chỉ chứa nhãn 0/1 và bao phủ đủ 1.225.601 target node: train 857.899, validation 183.862, test 183.840. Số fraud tương ứng là 10.857/2.326/2.326. | Split nguồn được giữ nguyên. Thống kê cũng phát hiện fraud chỉ chiếm khoảng 1,265% target node, báo trước rủi ro mất cân bằng cho Sprint 2. |

## 3. Vấn đề: Cần chọn cách xử lý missing value phù hợp với DGraphFin

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **3.1. Missing value được biểu diễn như thế nào trong DGraphFin?** | Đối chiếu Section 5.2 của paper với thống kê trực tiếp từng feature trong file NPZ; kiểm tra số lần xuất hiện `-1`, NaN và vô cực. | Paper xác nhận thiết lập Default thay missing value bằng `-1`. Trong dữ liệu, `-1` xuất hiện ở 16/17 feature; không có NaN hoặc vô cực. | `-1` được xác nhận là missing-value sentinel, không phải giá trị số thông thường cần suy đoán thêm. |
| **3.2. Nên giữ `-1` hay thêm missing indicator cho baseline?** | Ở Sprint 1, giữ thiết lập Default của paper: bảo toàn `-1` và chỉ chuyển `float64 → float32`. Đồng thời chuẩn bị biến thể zero-indicator: thay `-1` bằng 0 rồi nối thêm cờ missing cho từng chiều. | Hai biểu diễn được đưa vào protocol so sánh có kiểm soát ở Sprint 2: raw 17D và zero-indicator 34D. Zero-indicator tương ứng với Trick B trong Section 5.2 của paper. | Không tự chọn cách xử lý chỉ từ trực giác. Giữ raw 17D làm đối chứng và quyết định bằng validation AP; normalization hoặc imputation là các ablation riêng. |

## 4. Vấn đề: Cấu trúc và ngữ nghĩa graph cần được kiểm chứng trước khi xây GNN

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **4.1. Graph có hướng, self-loop, node cô lập và phân bố bậc như thế nào?** | Kiểm tra miền source/target, đếm self-loop và node cô lập, tính in-degree/out-degree và giữ thứ tự cạnh canonical `(source, target)`. | Graph có 4.300.999 cạnh có hướng, không có self-loop hoặc node cô lập; bậc vào tối đa 882, bậc ra tối đa 6, mean degree khoảng 1,162. | Hướng cạnh gốc được giữ nguyên trong canonical data. Không tự symmetrize vì làm vậy sẽ thay đổi protocol message passing và phải được đánh giá như một ablation riêng. |
| **4.2. Edge type và timestamp có đủ hợp lệ để dùng ở các sprint sau không?** | Kiểm tra chiều dài, miền giá trị và phân bố của `edge_type`/`edge_timestamp`; xác nhận mỗi thuộc tính căn đúng với một cạnh. | Có 11 edge type mã 1–11; timestamp có 821 giá trị trong miền 1–821; cả hai tensor đều có đúng 4.300.999 phần tử. | Hai thuộc tính được bảo toàn trong graph pipeline dù baseline Sprint 2 ban đầu là static và chưa khai thác 11 relation gốc. |

## 5. Vấn đề: Cần chuyển dữ liệu sang PyTorch Geometric mà không làm sai graph

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **5.1. Nên dùng `Data` hay `HeteroData`, và phải chuyển shape nào?** | Mô hình hóa toàn bộ user bằng một loại node trong PyG `Data`; chuyển `edge_index` từ `(E, 2)` sang `(2, E)`; gắn feature, label, edge type, timestamp và split index. | `x` có shape `(3.700.550, 17)`, `edge_index` có shape `(2, 4.300.999)`; toàn bộ tensor cần thiết được tạo thành công và contiguous. | `Data` là đủ vì dataset có một entity node là user; nhãn 2/3 là vai trò supervision khác, không phải một node type vật lý riêng bắt buộc dùng `HeteroData`. |
| **5.2. Full graph có thể được tải và chuyển đổi trên máy CPU mục tiêu không?** | Đo thời gian tải/chuyển `float32`, bộ nhớ các mảng canonical và thử full-data conversion sang tensor PyTorch CPU. | Tải dữ liệu và đổi feature mất khoảng 0,637 giây; các mảng canonical chiếm khoảng 408,8 MiB; full graph được chuyển sang tensor CPU thành công. | Data/graph conversion khả thi trong RAM mục tiêu, nhưng con số này chưa gồm activation, optimizer state và chi phí huấn luyện GNN. |

## 6. Vấn đề: Pipeline dữ liệu cần phát hiện lỗi và có khả năng tái kiểm tra

| Câu hỏi con cần giải quyết | Phương pháp (method) áp dụng | Đánh giá kết quả theo method | Kết luận/Ghi chú |
|---|---|---|---|
| **6.1. Validator có phát hiện được dữ liệu hoặc split không hợp lệ không?** | Viết unit/integration test trên NPZ tổng hợp cho loader, dtype conversion, validator, profiler và graph view; thêm test âm cố ý tạo split chồng lấn. | Test hợp lệ pass; test âm phát hiện đúng overlap. Full-data validation hoàn thành với 0 lỗi và 0 cảnh báo. | Không chỉ kiểm tra “pipeline chạy được”; validator còn kiểm tra schema, miền ID, nhãn, split, feature hữu hạn và độ dài thuộc tính cạnh. |
| **6.2. Kết quả profiling có thể tạo lại và kiểm tra độc lập không?** | Cung cấp CLI profiling; xuất báo cáo máy đọc được tại `artifacts/metrics/sprint1_data_profile.json`; lưu data dictionary và các quyết định xử lý dữ liệu. | Artifact chứa fingerprint, kết quả validation, thống kê node/cạnh/feature/label/split/degree/edge type/timestamp và có thể được sinh lại từ file nguồn. | Báo cáo Markdown dùng để giải thích; JSON profiling là bằng chứng máy đọc được cho các sprint và thí nghiệm tiếp theo. |

## Kết luận ngắn

- Sprint 1 đã chốt data contract, ý nghĩa nhãn, split và fingerprint của DGraphFin.
- Lớp 2/3 được xác định là background: giữ trong graph nhưng không thuộc supervised target 0/1.
- Paper xác nhận `-1` là missing-value sentinel; Sprint 1 giữ thiết lập Default làm đối chứng và chuẩn bị zero-indicator theo Trick B để đánh giá ở Sprint 2.
- Graph canonical giữ nguyên hướng cạnh, edge type và timestamp; mọi biến đổi như undirected phải là một ablation có kiểm soát.
- Full graph đã chuyển thành công sang PyTorch Geometric trên CPU, đồng thời validator và profiling artifact cho phép tái kiểm tra pipeline.
