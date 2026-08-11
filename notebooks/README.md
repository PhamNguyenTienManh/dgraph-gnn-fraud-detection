# Research notebooks

Ba notebook là luồng nghiên cứu chính và được đọc/chạy theo thứ tự:

1. `01_dgraphfin_eda.ipynb` — data contract, loader, validation, EDA và sampling;
2. `02_gnn_training.ipynb` — preprocessing, graph, GCN/GraphSAGE/RGCN và training;
3. `03_experiment_analysis.ipynb` — result catalog, multi-seed comparison và ablation.

Markdown trong notebook giải thích mục tiêu, giả định và kết luận; code được chia thành
các cell theo đúng thứ tự thực hiện.

Notebook 02 mặc định dùng `QUICK_MODE=True`. Quick mode chỉ là smoke test của pipeline.
Để chạy thí nghiệm chính thức, chọn `SELECTED_EXPERIMENT`, đặt
`RUN_OFFICIAL_EXPERIMENT=True` và chạy lại notebook. Raw run lưu ở
`artifacts/runs/` và được gitignore.

Notebook 03 mặc định đọc catalog đã track tại
`artifacts/metrics/sprint2_results.json`. Đặt
`REBUILD_CATALOG_FROM_LOCAL_RUNS=True` nếu muốn dựng lại catalog từ raw local runs.
