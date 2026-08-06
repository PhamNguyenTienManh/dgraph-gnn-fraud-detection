# DGraph Fraud-Ring Detection

Đồ án nghiên cứu ứng dụng **Graph Neural Networks (GNN)** vào bài toán phát hiện người dùng gian lận trên mạng lưới tài chính. Dự án sử dụng bộ dữ liệu **DGraphFin** và hướng tới khai thác cấu trúc liên kết giữa các tài khoản để hỗ trợ nhận diện các nhóm người dùng có dấu hiệu gian lận (fraud ring).

## Giới thiệu

Trong các hệ thống tài chính, hành vi gian lận không chỉ thể hiện qua thuộc tính của từng người dùng mà còn có thể xuất hiện trong mối quan hệ giữa nhiều tài khoản. Biểu diễn dữ liệu dưới dạng đồ thị cho phép mô hình học đồng thời:

- Đặc trưng của từng người dùng;
- Quan hệ giữa người dùng và người liên hệ khẩn cấp;
- Cấu trúc lân cận và hướng của liên kết;
- Vai trò của các nút nền không trực tiếp tham gia bài toán phân loại.

Đồ án xây dựng pipeline xử lý dữ liệu đồ thị, huấn luyện và so sánh các mô hình GCN, GraphSAGE, RGCN, GAT và TGAT trong điều kiện dữ liệu mất cân bằng mạnh. Các mô hình khai thác đặc trưng nút, cấu trúc liên kết, cơ chế attention và thông tin thời gian của đồ thị. Kết quả phân loại nút là nền tảng để tiếp tục phân tích cộng đồng và phát hiện fraud ring.

## Dữ liệu DGraphFin

DGraphFin là một đồ thị tài chính động trong thế giới thực, được giới thiệu tại NeurIPS 2022. Trong đồ thị:

- Mỗi nút đại diện cho một người dùng;
- Mỗi cạnh có hướng biểu diễn một người dùng khai báo người dùng khác làm liên hệ khẩn cấp;
- Mỗi nút có 17 đặc trưng đã được ẩn danh;
- Nhãn phân biệt người dùng bình thường, người dùng gian lận và các nút nền;
- Thời điểm cập nhật cạnh cung cấp thông tin về sự thay đổi của đồ thị theo thời gian.

Theo paper, DGraph gồm **3.700.550 nút**, **4.300.999 cạnh có hướng** và **1.225.601 nút có nhãn mục tiêu**. Trong số đó có 15.509 người dùng gian lận, khiến đây trở thành bài toán phân loại có mức độ mất cân bằng rất cao.

File dữ liệu `data/dgraphfin.npz` không được lưu trên GitHub vì dung lượng lớn. Sau khi tải dữ liệu, hãy đặt file vào thư mục `data/`.

## Nội dung thực hiện

Dự án hiện bao gồm:

- Kiểm định schema và thống kê dữ liệu DGraphFin;
- Xây dựng biểu diễn đồ thị có hướng bằng PyTorch Geometric;
- Xử lý giá trị thiếu và thử nghiệm đặc trưng zero-indicator;
- Huấn luyện và so sánh các mô hình GCN, GraphSAGE, RGCN, GAT và TGAT;
- Neighbor sampling để làm việc với đồ thị lớn trên tài nguyên giới hạn;
- Lưu cấu hình, metric, checkpoint và thông tin môi trường của từng lần chạy;
- Trực quan hóa và phân tích kết quả thực nghiệm.

## Cấu trúc thư mục

```text
dgraph-fraud-ring-detection/
├── configs/       # Cấu hình dữ liệu, mô hình và thí nghiệm
├── data/          # Dữ liệu cục bộ (không đưa dataset lớn lên GitHub)
├── docs/          # Tài liệu, kế hoạch và báo cáo thực nghiệm
├── src/           # Mã nguồn chính
├── tests/         # Kiểm thử tự động
├── pyproject.toml
└── README.md
```

## Công nghệ sử dụng

- Python 3.12
- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn
- Matplotlib
- pytest

## Chạy dự án

Tạo môi trường ảo và cài đặt project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[graph,train,dev,viz]"
```

Chạy kiểm thử:

```powershell
python -m pytest
```

Huấn luyện các mô hình baseline:

```powershell
python -m dgraph_fraud.cli.train_baselines --config configs/experiment/baseline_full.json
```

Sinh biểu đồ tổng hợp kết quả:

```powershell
python -m dgraph_fraud.cli.plot_sprint2
```

Các kết quả được sinh trong thư mục `artifacts/` và không được đưa lên GitHub.

## Tài liệu tham khảo

Đồ án được xây dựng với sự tham khảo chính từ:

> Xuanwen Huang et al. **DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection**. NeurIPS 2022, Datasets and Benchmarks Track.

- [Đọc paper DGraph](https://proceedings.neurips.cc/paper_files/paper/2022/file/8f1918f71972789db39ec0d85bb31110-Paper-Datasets_and_Benchmarks.pdf)
- [Trang giới thiệu bộ dữ liệu DGraph](https://dgraph.xinye.com/)

## Lưu ý

Dự án được thực hiện với mục đích học tập và nghiên cứu. Dữ liệu DGraphFin đã được ẩn danh; người sử dụng vẫn cần tuân thủ các điều khoản của đơn vị cung cấp dữ liệu.
