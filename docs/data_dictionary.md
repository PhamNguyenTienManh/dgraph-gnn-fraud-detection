# Từ điển dữ liệu DGraphFin

## Ngữ cảnh nghiệp vụ

- Node là một người dùng.
- Cạnh có hướng biểu diễn quan hệ giữa hai người dùng thông qua người liên hệ khẩn cấp.
- Đặc trưng và timestamp đã được ẩn danh.
- Bài toán nhãn ở Sprint 1–2 là phân loại node: bình thường (0) hoặc gian lận (1).
- Node lớp 2 và 3 là node nền: được giữ trong cấu trúc đồ thị nhưng không dùng để tính loss/metric phân loại nhị phân.

## Data contract

| Trường | Shape thực tế | Kiểu nguồn | Ý nghĩa/chính sách |
|---|---:|---|---|
| `x` | `(3.700.550, 17)` | `float64` | Đặc trưng node; chuyển sang `float32` trong bộ nhớ |
| `y` | `(3.700.550,)` | `int64` | Nhãn 0, 1, 2 hoặc 3 |
| `edge_index` | `(4.300.999, 2)` | `int64` | Mỗi hàng là `(source, target)` của cạnh có hướng |
| `edge_type` | `(4.300.999,)` | `int64` | Một trong 11 loại quan hệ đã ẩn danh |
| `edge_timestamp` | `(4.300.999,)` | `int64` | Thời điểm hình thành cạnh đã ẩn danh |
| `train_mask` | `(857.899,)` | `int64` | Chỉ số node train; tên trường gây hiểu nhầm, không phải boolean mask |
| `valid_mask` | `(183.862,)` | `int64` | Chỉ số node validation |
| `test_mask` | `(183.840,)` | `int64` | Chỉ số node test |

## Quy tắc toàn vẹn

- Node ID trong cạnh và split phải thuộc `[0, 3.700.550)`.
- Ba split không được chồng lấn và chỉ chứa node nhãn 0/1.
- Tổng ba split phải bao phủ các node nhãn 0/1 theo mô tả dataset.
- Chiều dài `edge_type` và `edge_timestamp` phải bằng số cạnh.
- Feature không được chứa NaN hoặc vô cực.
- Dữ liệu nguồn không bị ghi đè; mọi phép biến đổi phải có cấu hình và manifest.

## Quan sát về đặc trưng

- Feature có miền giá trị rất khác nhau; feature 10 có giá trị tối đa 1.313 trong khi nhiều feature chỉ nằm trong khoảng `[-1, 1]`.
- Giá trị `-1` xuất hiện nhiều ở 16/17 feature. Đây **có khả năng** là mã giá trị thiếu/không áp dụng, nhưng tài liệu nội bộ chưa xác nhận nên pipeline không tự thay thế hoặc chuẩn hóa giá trị này.
- Cấu hình Sprint 1 giữ nguyên giá trị feature và chỉ đổi kiểu từ `float64` sang `float32`. Việc chuẩn hóa hoặc thêm missing indicator phải được đánh giá bằng validation AP trong Sprint 2 và ghi như một biến thể thí nghiệm.

## Fingerprint nguồn

- SHA-256: `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85`
- Kích thước tệp: 680.317.982 byte.
