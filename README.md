# DGraph Fraud-Ring Detection

Đồ án nghiên cứu Graph Neural Networks cho bài toán phát hiện người dùng gian lận trên DGraphFin, hướng tới phân tích fraud-ring và đồ thị động ở các sprint sau.

Sprint 1 tập trung vào data contract, kiểm định dữ liệu, profiling, biểu diễn đồ thị có hướng và cấu hình sampling phù hợp máy CPU/16 GB RAM. Xem kế hoạch tại `docs/sprint1_sprint2_plan.md` và từ điển dữ liệu tại `docs/data_dictionary.md`.

Tệp dữ liệu nguồn được giữ nguyên tại `data/dgraphfin.npz`. Pipeline không ghi đè dữ liệu này.

## Lệnh kiểm tra Sprint 1

Sau khi kích hoạt `.venv` và cài project, chạy kiểm thử bằng `python -m pytest`. CLI `dgraph-profile` kiểm định/profiling toàn bộ dataset; module `dgraph_fraud.cli.benchmark_sampling` đo một batch neighbor sampling có hướng. Các báo cáo mặc định của sprint được lưu trong `artifacts/metrics/`.

Môi trường graph đã kiểm thử trên Windows CPU gồm PyTorch 2.12, PyG 2.8.0.post1 và `pyg-lib` 0.8.0. Wheel CPU của `pyg-lib` phải được cài từ trang wheel chính thức tương ứng với PyTorch 2.12 tại `data.pyg.org`.

## Sprint 2: GCN, GraphSAGE và RGCN

Chạy baseline raw 17 chiều:

```powershell
.\.venv\Scripts\python.exe -m dgraph_fraud.cli.train_baselines --config configs\experiment\baseline_full.json
```

Mỗi run lưu config, phiên bản môi trường, fingerprint dữ liệu, metric theo epoch, checkpoint tốt nhất và bảng so sánh trong `artifacts/runs/`. AP validation là tiêu chí duy nhất chọn checkpoint; test chỉ được tính sau khi tải lại trạng thái tốt nhất trong bộ nhớ.

Chạy biến thể zero-indicator 34 chiều:

```powershell
.\.venv\Scripts\python.exe -m dgraph_fraud.cli.train_baselines --config configs\experiment\baseline_full_zero_indicator.json
```

Không chạy đồng thời các cấu hình full nếu RAM/CPU của máy chưa được theo dõi. Kết quả ba seed và so sánh 17D/34D được ghi tại `docs/sprint2_report.md`.

Chạy RGCN 34 chiều với bốn relation target/background theo paper:

```powershell
.\.venv\Scripts\python.exe -m dgraph_fraud.cli.train_baselines --config configs\experiment\rgcn_background_full.json
```

RGCN dùng hidden size 13 để có 2.289 tham số, gần khớp GCN 34D có 2.305 tham số. Kết quả so sánh và giới hạn diễn giải về background node được ghi tại `docs/sprint2_report.md`.

## Biểu đồ Sprint 2

Cài dependency trực quan hóa và sinh lại các biểu đồ từ số liệu trong
`docs/sprint2_report.md`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[viz]"
.\.venv\Scripts\python.exe -m dgraph_fraud.cli.plot_sprint2
```

Mặc định, bốn file PNG được ghi vào `artifacts/figures/sprint2/`. Có thể đổi
thư mục và độ phân giải bằng `--output-dir` và `--dpi`.
