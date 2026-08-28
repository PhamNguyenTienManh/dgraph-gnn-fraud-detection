# DGraph Fraud Detection

Đồ án nghiên cứu ứng dụng **Graph Neural Networks (GNN)** vào bài toán phát hiện
người dùng gian lận trên mạng lưới tài chính DGraphFin.

Phần nghiên cứu được trình bày qua sáu Jupyter Notebook, đi từ khám phá dữ liệu,
huấn luyện và so sánh mô hình đến Community Detection, community-risk và giải thích
dự đoán bằng GNNExplainer. Người đọc có thể theo dõi lần lượt câu hỏi, phương pháp,
kết quả và giới hạn của từng thí nghiệm.

## Phạm vi hiện tại

Dự án đang giải bài toán phân loại fraud ở cấp node trong thiết lập
static/transductive:

- Kiểm định và khám phá dữ liệu DGraphFin;
- Xử lý missing sentinel và class imbalance;
- Huấn luyện và so sánh GCN, GraphSAGE, RGCN, GAT và TGAT;
- Neighbor sampling trên graph lớn;
- Time encoding `node_time - edge_time` và temporal attention cho TGAT undirected;
- Leiden Community Detection, risky-community discovery và visualization;
- Phân tích nhiều seed và community-feature ablation A/B/C/D;
- Giải thích TGAT + community-risk ở cấp node bằng GNNExplainer, sau đó kiểm tra lại
  lời giải thích bằng các phép tác động trực tiếp lên feature và event.

Community được dùng để khoanh vùng ứng viên rủi ro, không phải xác nhận rằng các node
trong community có hành vi gian lận phối hợp. Kết quả model vẫn là transductive node classification.

## Thứ tự đọc và chạy

1. [`notebooks/01_dgraphfin_eda.ipynb`](notebooks/01_dgraphfin_eda.ipynb) — load,
   validation, EDA và benchmark neighbor sampling;
2. [`notebooks/02_gnn_training.ipynb`](notebooks/02_gnn_training.ipynb) — feature
   preprocessing, PyG graph, model definitions, training và lưu raw artifact;
3. [`notebooks/03_experiment_analysis.ipynb`](notebooks/03_experiment_analysis.ipynb)
   — tổng hợp nhiều seed, ablation, trực quan hóa và kết luận;
4. [`notebooks/04_community_detection.ipynb`](notebooks/04_community_detection.ipynb)
   — Leiden Community Detection, EDA cấp community và tạo community feature;
5. [`notebooks/05_community_ablation.ipynb`](notebooks/05_community_ablation.ipynb)
   — kiểm tra structural community features và community-risk có giúp TGAT trên ba seed hay không.
6. [`notebooks/06_gnn_explainer.ipynb`](notebooks/06_gnn_explainer.ipynb)
   — xem TGAT + community-risk dựa vào feature và event nào, đồng thời đánh giá lời
   giải thích có đủ ổn định và đáng tin để sử dụng hay không.

Các notebook đã lưu sẵn output của lần chạy kiểm tra gần nhất. Notebook training mặc
định dùng `QUICK_MODE=True` với ít epoch/batch để kiểm tra pipeline; kết quả quick mode
không được dùng làm kết luận nghiên cứu.

## Cấu trúc dự án

```text
dgraph-fraud-detection/
├── artifacts/
│   ├── figures/       # Hình dùng trong báo cáo
│   ├── metrics/       # Profile và result catalog có thể kiểm tra độc lập
│   └── runs/          # Checkpoint/output thô tại local, được gitignore
├── data/              # Dataset DGraphFin tại local
├── docs/              # Báo cáo và kế hoạch của các sprint
├── notebooks/         # Luồng trình bày và thực thi các thí nghiệm
├── scripts/           # Code hỗ trợ chạy, tổng hợp và kiểm tra kết quả
├── requirements.txt
└── README.md
```

## Cài đặt môi trường

Yêu cầu Python 3.12. Tạo virtual environment và cài dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dgraph-fraud --display-name "Python (DGraph Fraud)"
```

`NeighborLoader` cần sampling backend tương thích với phiên bản PyTorch/PyG của môi
trường, chẳng hạn `pyg-lib`. Nếu môi trường chưa có backend này, cài wheel tương thích
theo hướng dẫn của PyTorch Geometric trước khi chạy notebook 02.

Khởi động JupyterLab:

```powershell
python -m jupyter lab
```

Mở notebook theo thứ tự `01 → 02 → 03 → 04 → 05 → 06`. Chọn kernel
`Python (DGraph Fraud)`.

## Chạy full experiment

Trong `02_gnn_training.ipynb`:

1. Chọn tên trong `SELECTED_EXPERIMENT`;
2. Giữ cấu hình chính thức trong `EXPERIMENTS`;
3. Đặt `RUN_OFFICIAL_EXPERIMENT=True`;
4. Chạy lại notebook.

Raw output được lưu theo timestamp trong `artifacts/runs/`. Để dựng lại catalog từ
các raw run, đặt `REBUILD_CATALOG_FROM_LOCAL_RUNS=True` trong notebook 03.

Kết quả full GAT/TGAT và phần diễn giải được tổng hợp trong
[`docs/sprint3_report.md`](docs/sprint3_report.md).

Kết quả Community Detection, fraud concentration và so sánh A/B/C/D được tổng hợp
trong [`docs/sprint4_report.md`](docs/sprint4_report.md). Notebook 05 mặc định chỉ hiển thị kết quả
đã lưu. Dùng `RUN_COMMUNITY_ABLATION_TRAIN=1` để train/reuse ba seed và chỉ dùng
`RUN_COMMUNITY_ABLATION_TEST=1` cho lần đánh giá test đã khóa.

Sprint 4 chọn **TGAT + community-risk** vì mô hình này có validation AP tốt nhất
trong các cấu hình đã so sánh. Sprint 5 dùng GNNExplainer để tìm hiểu mô hình này chú
ý đến thông tin nào. Kết quả cho thấy các xu hướng feature có thể dùng để hiểu model ở
cấp nhóm node, nhưng thứ hạng event chưa vượt rõ các cách chọn đơn giản. Vì vậy Sprint
5 chưa dùng lời giải thích event làm bằng chứng điều tra. Xem phần diễn giải đầy đủ tại
[`docs/sprint5_report.md`](docs/sprint5_report.md) và bản tóm tắt thiết kế thí nghiệm tại
[`docs/sprint5_problem_method_evaluation.md`](docs/sprint5_problem_method_evaluation.md).

## Artifact policy

`artifacts/runs/` được ignore vì chứa checkpoint lớn và output theo timestamp. Các
metric đã chuẩn hóa trong `artifacts/metrics/` và hình trong `artifacts/figures/` được
track để kết quả thực nghiệm có thể được kiểm tra mà không cần chạy lại toàn bộ experiment.

## Tài liệu tham khảo

> Xuanwen Huang et al. **DGraph: A Large-Scale Financial Dataset for Graph Anomaly
> Detection**. NeurIPS 2022, Datasets and Benchmarks Track.

- [DGraph paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/8f1918f71972789db39ec0d85bb31110-Paper-Datasets_and_Benchmarks.pdf)
- [DGraph dataset website](https://dgraph.xinye.com/)

## Lưu ý

Dự án được thực hiện cho mục đích học tập và nghiên cứu. Dữ liệu đã được ẩn danh;
người sử dụng vẫn cần tuân thủ điều khoản của đơn vị cung cấp dữ liệu.
