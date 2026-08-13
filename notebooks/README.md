# Research notebooks

Ba notebook là luồng nghiên cứu chính và được đọc/chạy theo thứ tự:

1. `01_dgraphfin_eda.ipynb` — data contract, loader, validation, EDA và sampling;
2. `02_gnn_training.ipynb` — preprocessing, graph, GCN/GraphSAGE/RGCN/GAT/TGAT
   và training;
3. `03_experiment_analysis.ipynb` — result catalog, multi-seed comparison và ablation.

Markdown trong notebook giải thích mục tiêu, giả định và kết luận; code được chia thành
các cell theo đúng thứ tự thực hiện.

Notebook 02 mặc định dùng `QUICK_MODE=True`. Kết quả quick mode chỉ dùng kiểm tra
kỹ thuật, không dùng để so sánh chất lượng model. Muốn chủ động chạy lại thí
nghiệm, chọn `SELECTED_EXPERIMENT`, đặt `RUN_OFFICIAL_EXPERIMENT=True` và chạy lại
notebook. Raw run lưu ở `artifacts/runs/` và được gitignore.

TGAT mặc định dùng graph undirected, fan-out `[15]`, `node_time - edge_time`, cosine
time encoding và một PyG `TransformerConv`. Cấu hình này đã chạy full đủ seed
42/43/44; notebook vẫn giữ quick mode và tắt cờ official để tránh vô tình chạy lại.

Notebook 03 đọc catalog baseline tại `artifacts/metrics/sprint2_results.json` và
catalog GAT/TGAT tại `artifacts/metrics/sprint3_results.json`. Đặt
`REBUILD_CATALOG_FROM_LOCAL_RUNS=True` nếu muốn dựng lại catalog baseline từ raw
local runs.
