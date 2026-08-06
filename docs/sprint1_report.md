# Báo cáo Sprint 1 — Data và Graph Pipeline

## Trạng thái

Pipeline dữ liệu cốt lõi đã được triển khai và kiểm thử. Báo cáo profiling máy đọc được nằm tại `artifacts/metrics/sprint1_data_profile.json` và được tạo lại bằng CLI của project.

## Kết quả kiểm định dataset

| Hạng mục | Kết quả |
|---|---:|
| SHA-256 | `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85` |
| Node | 3.700.550 |
| Cạnh có hướng | 4.300.999 |
| Chiều feature | 17 |
| Normal (0) | 1.210.092 |
| Fraud (1) | 15.509 |
| Background (2) | 1.620.851 |
| Background (3) | 854.098 |
| Train | 857.899 (10.857 fraud) |
| Validation | 183.862 (2.326 fraud) |
| Test | 183.840 (2.326 fraud) |
| Loại cạnh | 11, mã từ 1 đến 11 |
| Timestamp | 821 giá trị, miền 1–821 |
| Self-loop | 0 |
| Node cô lập | 0 |
| Bậc vào tối đa | 882 |
| Bậc ra tối đa | 6 |

Validator xác nhận schema hợp lệ, ba split không chồng lấn, chỉ chứa nhãn 0/1 và bao phủ toàn bộ node thuộc hai lớp dự đoán.

## Quyết định kỹ thuật

- Giữ nguyên hướng cạnh `(source, target)`; không tự động symmetrize.
- Giữ lớp 2/3 trong đồ thị để truyền thông tin, nhưng loại khỏi loss và metric.
- Chuyển `x` từ `float64` sang `float32` trong bộ nhớ.
- Không chuẩn hóa feature mặc định vì ý nghĩa feature đã ẩn danh và `-1` có thể là missing sentinel chưa được xác nhận.
- Bảo toàn `edge_type` và `edge_timestamp` để dùng cho các sprint sau.
- Trên máy CPU/16 GB RAM, khởi đầu bằng neighbor sampling hai lớp với fan-out `[15, 10]`, batch size 1.024 và `num_workers=0` trên Windows.
- CSR sampler thuần NumPy lấy lân cận đi vào theo luồng message `source -> target`; không phụ thuộc native extension và không nhân đôi cạnh.
- Không dùng full-batch làm cấu hình huấn luyện mặc định.

## Benchmark sampling trên dữ liệu thật

Benchmark được chạy với batch 1.024 node train, fan-out `[15, 10]`, seed 42 và sampling lân cận đi vào.

| Hạng mục | Kết quả lần đo |
|---|---:|
| Tải dataset và đổi feature sang `float32` | 0,637 giây |
| Dựng CSR index | 0,570 giây |
| Sample một batch | 0,032 giây |
| Bộ nhớ các mảng canonical | 428.678.576 byte (~408,8 MiB) |
| Bộ nhớ CSR index | 64.012.400 byte (~61,0 MiB) |
| Node trong batch sampled | 3.207 |
| Cạnh trong batch sampled | 2.341 |
| Cạnh theo hop | 1.282 / 1.059 |

Đây là benchmark pipeline/sampling, chưa bao gồm tensor trung gian, optimizer state hoặc activation khi huấn luyện. Cần để dư RAM đáng kể cho Sprint 2.

## Kiểm thử

- Unit/integration test trên NPZ tổng hợp kiểm tra loader, chuyển `float32`, validator, profiler và graph view.
- Có test âm cho split chồng lấn.
- Full-data validation đã hoàn thành không lỗi và không cảnh báo.
- Full-data conversion sang tensor PyTorch CPU đã thành công; `x` có shape `(3.700.550, 17)`, `edge_index` có shape `(2, 4.300.999)` và các tensor đều contiguous.

## Cập nhật môi trường trước Sprint 2

- Đã đồng bộ môi trường về PyTorch `2.12.0+cpu`, PyG `2.8.0.post1` và `pyg-lib 0.8.0+pt212cpu` từ các kho chính thức.
- `NeighborLoader` đã chạy thành công trên DGraph thật với batch size 1.024 và fan-out `[15, 10]`: 3.207 node, 2.337 cạnh, batch đầu khoảng 0,004 giây và RSS tiến trình khoảng 835 MiB.
- PyG `NeighborLoader` là sampler chính của Sprint 2; CSR sampler thuần NumPy được giữ làm phương án fallback và đối chiếu.
- Cần chốt việc message passing sử dụng cạnh theo hướng gốc, cạnh đảo hay hai chiều. Dữ liệu canonical luôn giữ hướng gốc; biến thể hai chiều chỉ là ablation.
- Cần đo validation AP cho raw feature so với phương án xử lý missing/normalization; không dùng test để chọn.
