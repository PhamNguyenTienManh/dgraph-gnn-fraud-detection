# Research notebooks

Bảy notebook là luồng nghiên cứu chính và được đọc/chạy theo thứ tự:

1. `01_dgraphfin_eda.ipynb` — data contract, loader, validation, EDA và sampling;
2. `02_gnn_training.ipynb` — preprocessing, graph, GCN/GraphSAGE/RGCN/GAT/TGAT
   và training;
3. `03_experiment_analysis.ipynb` — result catalog, multi-seed comparison và ablation;
4. `04_community_detection.ipynb` — Community Detection, community-level EDA và
   tạo community feature cho Sprint 4;
5. `05_community_ablation.ipynb` — kiểm tra structural community features và
   community-risk có giúp TGAT trên ba seed hay không;
6. `06_gnn_explainer.ipynb` — giải thích 80 node bằng GNNExplainer, kiểm tra trực tiếp
   event/feature nào tác động đến dự đoán và đánh giá lời giải thích có đủ tin cậy để
   đề xuất graph con hay không;
7. `07_tgat_error_analysis.ipynb` — khóa threshold từ validation, phân tích
   feature/community-risk trên error-control đã ghép và chạy full-split TGAT input ablation.

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

Notebook 04 dùng Leiden qua `igraph`, chạy partition toàn graph, community EDA và tạo
structural/train-only risk features. Notebook 05 tái sử dụng pipeline TGAT từ notebook
02. Dùng `RUN_COMMUNITY_ABLATION_TRAIN=1` để train/reuse B/C/D đủ ba seed và chỉ dùng
`RUN_COMMUNITY_ABLATION_TEST=1` cho lần đánh giá test đã khóa; mặc định cả hai đều tắt.

Notebook 06 đọc kết quả cuối đã lưu và trình bày bốn hình: quy mô neighborhood,
feature thường được chọn, tác động của community-risk và kiểm tra lời giải thích event.
Các kết quả graph mô tả phép tính của model, không phải fraud ring đã được xác nhận hay
danh sách ưu tiên điều tra.

Notebook 07 mặc định đọc các artifact đã khóa. Có thể chạy lại từng phần bằng
`RUN_SPRINT5_2_PREPARATION=1`, `RUN_ERROR_ATTRIBUTION=1` hoặc `RUN_TGAT_ABLATION=1`;
không bật các cờ này khi chỉ cần đọc kết quả đã kiểm chứng.
