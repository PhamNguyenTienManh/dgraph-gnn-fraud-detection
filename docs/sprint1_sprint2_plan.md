# Kế hoạch triển khai Sprint 1 và Sprint 2

## 1. Phạm vi và nguồn thông tin

Tài liệu này chỉ lập kế hoạch cho hai sprint đầu của đề tài **“Graph Neural Networks for Dynamic Fraud-Ring Detection on the DGraph Dataset”**. Tài liệu không bao gồm hoạt động triển khai mã nguồn, tạo môi trường, cài thư viện hoặc huấn luyện mô hình.

### 1.1. Nguồn đã đối chiếu

- Nội dung Sprint 1 và Sprint 2 trong `docs/plan.md`.
- Mô tả dữ liệu trong `data/Readme.md`.
- Tệp dữ liệu hiện có: `data/dgraphfin.npz`.


### 1.2. Thông tin đã được xác nhận từ tài liệu nội bộ

| Hạng mục | Thông tin |
|---|---|
| Đặc trưng node | `x`, gồm 17 chiều |
| Nhãn node | `y`, gồm 4 lớp |
| Lớp dùng để dự đoán | Lớp 0: người dùng bình thường; lớp 1: người dùng gian lận |
| Lớp nền | Lớp 2 và lớp 3 |
| Số node theo lớp | Lớp 0: 1.210.092; lớp 1: 15.509; lớp 2: 1.620.851; lớp 3: 854.098 |
| Cạnh | `edge_index`, kích thước `(4.300.999, 2)` |
| Loại cạnh | `edge_type`, gồm 11 loại |
| Thời gian cạnh | `edge_timestamp`, thời gian đã được ẩn danh |
| Chia tập | `train_mask`, `valid_mask`, `test_mask`; các node lớp 0 và 1 được chia ngẫu nhiên theo tỷ lệ 70/15/15 |
| Kiểu split thực tế | Ba trường có tên `*_mask` thực tế là mảng chỉ số node kiểu `int64`, không phải boolean mask |
| Hướng đồ thị | Có hướng: cạnh biểu diễn quan hệ người dùng cung cấp người liên hệ khẩn cấp |

## 2. Phân tích mục tiêu

### 2.1. Sprint 1 — Chuẩn bị dữ liệu và Graph Pipeline

Mục tiêu của Sprint 1 là tạo một nền dữ liệu **đúng, kiểm chứng được, tái lập được và vừa với tài nguyên máy cá nhân** trước khi huấn luyện. Sprint này cần trả lời bốn nhóm câu hỏi:

1. Dataset biểu diễn node, cạnh, loại cạnh, thời gian, đặc trưng và nhãn như thế nào?
2. Dữ liệu nào được phép dùng cho train, validation và test?
3. Đồ thị phải được chuyển thành dạng nào để GCN và GraphSAGE sử dụng được?
4. Nên dùng toàn đồ thị, mini-batch theo lân cận hay subgraph cố định để tránh vượt tài nguyên?

Thành công của Sprint 1 không chỉ là “đọc được tệp”, mà là có pipeline tạo ra cùng một dữ liệu đầu vào khi chạy lại, có báo cáo kiểm định số node/cạnh/chiều đặc trưng/phân bố nhãn, và không gây rò rỉ dữ liệu ngoài chủ đích.

### 2.2. Sprint 2 — Baseline GCN và GraphSAGE

Mục tiêu của Sprint 2 là xây dựng một khung thực nghiệm thống nhất để huấn luyện, validation, inference và so sánh GCN với GraphSAGE. Hai mô hình phải dùng cùng dữ liệu, split, quy tắc tiền xử lý, tiêu chí chọn checkpoint và ngân sách thực nghiệm; các khác biệt cần thiết do kiến trúc phải được ghi lại.

Do lớp gian lận rất hiếm so với lớp bình thường, **Average Precision (AP)** là chỉ số chính cần quan sát cùng với **ROC-AUC**. Sprint này cũng phải lưu được cấu hình, checkpoint, log và kết quả để các sprint sau dùng làm benchmark.

### 2.3. Ranh giới cần làm rõ

`plan.md` mô tả GCN và GraphSAGE ở Sprint 2 nhưng chưa yêu cầu khai thác thời gian trực tiếp; mô hình temporal được dành cho sprint sau. Vì vậy:

- Sprint 1 phải bảo toàn `edge_timestamp` và kiểm tra chất lượng trường này.
- Sprint 2 được xem là **baseline đồ thị tĩnh**; timestamp chưa đi vào message passing, trừ khi được phê duyệt bổ sung.
- Nhãn hiện có hỗ trợ bài toán phân loại node gian lận/không gian lận. Tiêu chí xác định một **fraud ring**, nhãn cấp ring và chỉ số đánh giá cấp ring chưa được nêu. Vì vậy Sprint 1–2 chưa tuyên bố giải quyết đầy đủ phát hiện fraud-ring cấp nhóm.

## 3. Các giả định và điểm cần phê duyệt

| Mã | Giả định/điểm chưa rõ | Phương án lập kế hoạch tạm thời |
|---|---|---|
| A1 | Chưa nêu framework GNN | Dùng PyTorch và PyTorch Geometric (PyG). |
| A2 | Phiên bản Python và phần cứng đã khảo sát | Dùng Python 3.12.1, PyTorch 2.12 CPU, PyG 2.8.0.post1 và `pyg-lib` 0.8.0. Máy không có NVIDIA/CUDA. |
| A3 | Đồ thị đã được xác nhận là có hướng | Giữ nguyên cạnh có hướng ở bản dữ liệu chuẩn; chỉ tạo biến thể hai chiều như một ablation riêng nếu được phê duyệt. |
| A4 | Chưa nói cách dùng lớp 2 và 3 | Cho phép lớp nền tham gia truyền thông tin trên đồ thị, nhưng chỉ tính loss và metric trên node lớp 0/1 thuộc đúng mask. |
| A5 | Split được cung cấp là ngẫu nhiên, chưa phải temporal split | Giữ split 70/15/15 có sẵn để đúng phạm vi baseline; ghi rõ hạn chế và không tự tạo temporal split. |
| A6 | Chưa nêu chuẩn hóa đặc trưng | Mặc định không chuẩn hóa. Biến thể `global_zscore` chỉ là ablation transductive được ghi nhãn rõ; không trộn kết quả của nó với protocol inductive. |
| A7 | Chưa nêu chiến lược mất cân bằng | So sánh tối thiểu cấu hình không trọng số với `class weight` hoặc sampling cân bằng; chọn bằng validation AP, không dùng test. |
| A8 | Máy có 15,47 GB RAM, CPU Intel Core Ultra 7 155U (12 core/14 luồng), Intel Graphics và không có CUDA | Mặc định CPU + neighbor sampling; `batch_size=1024`, fan-out `[15, 10]`, `num_workers=0` trên Windows là cấu hình khởi đầu và sẽ điều chỉnh bằng benchmark. Không dùng full-batch mặc định. |
| A9 | Chưa quy định số seed/số lần chạy | Dùng tối thiểu 3 seed cho kết quả chính. |
| A10 | Chưa định nghĩa “cùng điều kiện” tuyệt đối | Cố định dữ liệu, split, metric, seed, số epoch tối đa, early stopping và cách chọn checkpoint; cho phép tham số kiến trúc khác nhau nhưng phải công khai. |

## 4. Cấu trúc thư mục

```text
dgraph-fraud-ring-detection/
├── data/
│   ├── dgraphfin.npz
│   ├── Readme.md
│   ├── processed/             # Artifact đã xử lý, không đưa dữ liệu lớn vào Git
│   └── samples/               # Subgraph/sampling artifact nếu cần
├── docs/
│   ├── sprint1_sprint2_plan.md
│   ├── data_dictionary.md
│   └── experiment_protocol.md
├── notebooks/
│   ├── 01_dgraphfin_eda.ipynb
│   ├── 02_gnn_training.ipynb
│   └── 03_experiment_analysis.ipynb
├── artifacts/
│   ├── metrics/
│   ├── figures/
│   └── runs/                  # Checkpoint/output timestamp ở local
├── README.md
└── requirements.txt
```

### Nguyên tắc tổ chức

- `data/` chứa dữ liệu nguồn và dữ liệu dẫn xuất; dữ liệu lớn không đưa vào version control.
- Notebook chứa trực tiếp các hàm, class và cấu hình research; code được chia thành cell ngắn theo đúng luồng phân tích.
- Registry cấu hình thí nghiệm nằm trong notebook training để toàn bộ biến kiểm soát được trình bày tập trung trong cùng một luồng thực nghiệm.
- Mỗi lần chạy có mã định danh riêng và lưu cấu hình, seed, metric, log, checkpoint.
- `artifacts/` là đầu ra tái tạo được; chỉ các báo cáo nhỏ cần thiết mới cân nhắc quản lý bằng Git.

## 5. Công nghệ và thư viện dự kiến

| Nhóm | Công nghệ/thư viện | Mục đích |
|---|---|---|
| Ngôn ngữ | Python 3.12.1 | Xử lý dữ liệu và thực nghiệm ML |
| Deep learning | PyTorch | Tensor, tối ưu hóa, huấn luyện, checkpoint |
| Graph learning | PyTorch Geometric | Biểu diễn graph, GCN, GraphSAGE, neighbor sampling |
| Dữ liệu số | NumPy | Đọc và kiểm tra tệp NPZ |
| Bảng/báo cáo | pandas | Tổng hợp thống kê và kết quả thực nghiệm |
| Metric | scikit-learn | ROC-AUC, Average Precision, confusion matrix bổ trợ |
| Cấu hình | Dictionary/registry hiển thị trực tiếp trong notebook | Quản lý tham số có kiểm soát và dễ review |
| Theo dõi | TensorBoard và tệp CSV/JSON cục bộ | Theo dõi loss, metric, thời gian và cấu hình |
| Trực quan | Matplotlib, Seaborn | Phân bố nhãn, learning curve, so sánh mô hình |
| Kiểm tra | Assert, validation cell và quick smoke run | Dừng sớm khi dữ liệu hoặc pipeline không hợp lệ |
| Chất lượng | Hàm/class ngắn, docstring, tên rõ nghĩa và cell theo một trách nhiệm | Giữ notebook dễ đọc và dễ review |
| Tài nguyên | psutil; công cụ CUDA của PyTorch nếu có GPU | Theo dõi RAM/VRAM và thời gian chạy |

> Các dependency đã dùng để chạy notebook được khóa theo khoảng/version trong `requirements.txt`; PyTorch/PyG sampling backend vẫn phải tương thích với hệ điều hành và kiến trúc CPU/GPU.

### Cấu hình máy mục tiêu và hệ quả thiết kế

| Thành phần | Kết quả khảo sát | Hệ quả |
|---|---|---|
| Máy/OS | HP ProBook 440 G11, Windows 11 Pro 64-bit | `num_workers=0` là mặc định an toàn cho DataLoader trên Windows; chỉ tăng sau benchmark |
| CPU | Intel Core Ultra 7 155U, 12 core/14 luồng | Có thể xử lý pipeline NumPy và train mini-batch, nhưng thời gian huấn luyện GNN sẽ dài hơn GPU |
| RAM | 15,47 GB | Đủ tải dữ liệu thô khoảng 680 MB, nhưng cần tránh nhiều bản sao tensor và đóng ứng dụng nặng khi train |
| GPU | Intel integrated graphics; không có NVIDIA/CUDA | PyTorch/PyG chạy CPU; không lập kế hoạch dựa vào CUDA |
| Ổ C | Khoảng 397 GB trống tại thời điểm khảo sát | Đủ artifact, nhưng không tự động nhân bản nhiều bản dataset/checkpoint |

`x` có kiểu `float64` và chiếm khoảng 503 MB; pipeline chuyển sang `float32` một lần để giảm khoảng một nửa bộ nhớ. `edge_index` được giữ `int64` theo yêu cầu chỉ số của PyTorch/PyG. Vì NPZ hiện không nén, có thể cân nhắc cache/memory-map ở bước tối ưu sau, nhưng không tạo thêm bản sao 680 MB nếu chưa chứng minh cần thiết.

> **Cập nhật trước Sprint 2:** PyTorch 2.12 CPU, PyG 2.8.0.post1 và `pyg-lib` 0.8.0 đã cài thành công. `NeighborLoader` chạy được trên DGraph thật và được chọn làm sampler chính; CSR sampler đã kiểm thử được giữ làm fallback. Việc thay backend không làm thay đổi data contract hoặc split.

## 6. Các khối code cần xây dựng trong notebook

### 6.1. Các khối code Sprint 1

| Khối code/cell | Trách nhiệm chính | Đầu ra |
|---|---|---|
| Data schema/contract | Định nghĩa key bắt buộc, shape, dtype, ý nghĩa nhãn và mask | Data contract có phiên bản |
| NPZ loader | Đọc dữ liệu có kiểm soát bộ nhớ; báo lỗi rõ khi thiếu/sai key | Cấu trúc dữ liệu thô |
| Data validator | Kiểm tra shape, chỉ số cạnh, NaN/Inf, mask chồng lấn, nhãn, timestamp, edge type | Báo cáo validation pass/fail |
| Dataset profiler | Thống kê node, cạnh, degree, nhãn, split, feature, loại cạnh, timestamp | Bảng và biểu đồ EDA |
| Preprocessor | Chuẩn hóa/biến đổi đặc trưng theo thống kê train; giữ metadata | Dữ liệu đã xử lý + tham số biến đổi |
| Graph builder | Chuyển sang định dạng PyG; quy định hướng cạnh, self-loop và kiểu dữ liệu | Graph artifact |
| Sampler/subgraph builder | Neighbor sampling hoặc subgraph tái lập theo seed | Cấu hình sampler/artifact mẫu |
| Split manager | Dùng mask có sẵn và bảo vệ ranh giới train/valid/test | Split metadata |
| Reproducibility utility | Quản lý seed, fingerprint dữ liệu và cấu hình | Manifest cho mỗi lần chạy |

### 6.2. Các khối code Sprint 2

| Khối code/cell | Trách nhiệm chính | Đầu ra |
|---|---|---|
| Model interface | Chuẩn hóa input/output và cấu hình mô hình | Giao diện chung cho baseline |
| GCN baseline | Baseline convolution đồ thị | Logit/xác suất gian lận |
| GraphSAGE baseline | Baseline tổng hợp lân cận | Logit/xác suất gian lận |
| Loss/imbalance strategy | Binary loss và tùy chọn class weight/sampling | Loss nhất quán |
| Trainer | Train theo mini-batch/full-batch, early stopping, gradient clipping nếu cần | Checkpoint và training log |
| Validator | Tính validation loss, AUC, AP; chọn checkpoint | Best checkpoint theo quy tắc đã chốt |
| Inference | Chạy dự đoán theo batch, không trộn dữ liệu test vào chọn mô hình | Prediction artifact |
| Evaluator | AUC/AP và metric bổ trợ; bootstrap/độ lệch giữa seed nếu đủ tài nguyên | Báo cáo metric |
| Experiment runner | Nạp cấu hình, tạo run ID, điều phối và lưu provenance | Thư mục run hoàn chỉnh |
| Comparison reporter | Bảng so sánh GCN/GraphSAGE, learning curve, tài nguyên | Báo cáo baseline |

## 7. Luồng xử lý dữ liệu (Data Pipeline)

### 7.1. Pipeline chuẩn của Sprint 1

1. **Kiểm kê nguồn:** xác nhận checksum/fingerprint, dung lượng, key và tài liệu mô tả của `dgraphfin.npz`.
2. **Đọc metadata trước:** kiểm tra key, shape, dtype và ước lượng RAM trước khi materialize các mảng lớn.
3. **Kiểm định schema:** đối chiếu `x`, `y`, `edge_index`, `edge_type`, `edge_timestamp` và ba mask.
4. **Kiểm định toàn vẹn:** bảo đảm node ID của cạnh nằm trong miền hợp lệ; chiều dài edge type/timestamp khớp số cạnh; mask đúng kiểu, không chồng lấn ngoài dự kiến; không có NaN/Inf bất thường.
5. **Lập hồ sơ dữ liệu:** thống kê phân bố nhãn/split, degree, isolated node, self-loop, cạnh trùng, loại cạnh và miền timestamp.
6. **Xác lập tập dự đoán:** lớp 0/1 là đối tượng tính loss/metric; lớp 2/3 được xử lý theo A4 và không tự gán thành lớp âm.
7. **Tiền xử lý đặc trưng:** fit mọi tham số có học từ dữ liệu trên train, lưu tham số rồi áp dụng cho validation/test.
8. **Tạo graph representation:** giữ bản canonical gần dữ liệu gốc; các biến đổi như hai chiều/self-loop là bước có cấu hình, không ghi đè dữ liệu gốc.
9. **Chọn sampling:** benchmark nhanh mức dùng RAM/VRAM và thời gian; chọn neighbor sampling hoặc subgraph phù hợp; seed và quy tắc sampling phải được lưu.
10. **Lưu manifest:** ghi fingerprint dữ liệu, cấu hình, thống kê và kết quả validation để tái lập.

### 7.2. Pipeline train/evaluate của Sprint 2

`Dữ liệu đã kiểm định → cấu hình chung → sampler/dataloader → GCN hoặc GraphSAGE → loss trên node train lớp 0/1 → validation AUC/AP → chọn checkpoint → inference test một lần → tổng hợp nhiều seed → báo cáo so sánh`

Quy tắc chống rò rỉ:

- Không dùng nhãn validation/test để fit scaler, sampler có giám sát hoặc class weight.
- Không dùng test để chọn epoch, hyperparameter hoặc threshold.
- Nếu message passing trên đồ thị transductive nhìn thấy cấu trúc/cạnh của node validation/test, phải ghi rõ đây là **thiết lập transductive**; tuyệt đối không dùng nhãn của các node đó trong train.
- Mọi thay đổi về cạnh theo thời gian hoặc temporal cutoff nằm ngoài baseline hiện tại nếu chưa được phê duyệt.

## 8. Kế hoạch công việc Sprint 1

Đề xuất thời lượng: **2 tuần (10 ngày làm việc)**; có thể điều chỉnh sau khi biết nguồn lực và cấu hình máy.

| Mốc | Thời gian dự kiến | Công việc | Tiêu chí hoàn thành |
|---|---:|---|---|
| S1-M1: Data contract | Ngày 1–2 | Đọc tài liệu, kiểm kê key/shape/dtype, chốt ý nghĩa lớp và mask, lập danh sách giả định | Data dictionary và contract được review |
| S1-M2: Validation & profiling | Ngày 3–4 | Thiết kế loader, kiểm tra toàn vẹn, thống kê feature/label/split/edge/timestamp | Báo cáo validation không còn lỗi mức blocker |
| S1-M3: Graph representation | Ngày 5–6 | Chốt hướng cạnh, lớp nền, self-loop, chuẩn hóa và format PyG | Quyết định có tài liệu; graph artifact tái lập được |
| S1-M4: Sampling & resource test | Ngày 7–8 | Ước lượng RAM/VRAM; thử cấu hình sampling nhỏ; đánh giá coverage và tốc độ | Có cấu hình chạy được trên máy mục tiêu |
| S1-M5: Kiểm tra & bàn giao | Ngày 9–10 | Chạy notebook đầu-cuối, kiểm tra assert/validation, hoàn thiện artifact và báo cáo | Checklist nghiệm thu Sprint 1 đạt |

### Checklist nghiệm thu Sprint 1

- [ ] Schema và ý nghĩa tất cả trường đã được ghi lại.
- [ ] Số node, số cạnh, 17 chiều đặc trưng và phân bố bốn lớp được đối chiếu.
- [ ] Ba mask được kiểm tra về tỷ lệ, giao nhau và miền nhãn.
- [ ] `edge_type` và `edge_timestamp` được bảo toàn.
- [ ] Quyết định về hướng cạnh, self-loop và lớp nền được phê duyệt.
- [ ] Không có bước fit tiền xử lý sử dụng validation/test.
- [ ] Sampling chạy được trong giới hạn RAM/VRAM đã đo.
- [ ] Pipeline có seed, fingerprint và manifest để tái lập.
- [ ] Notebook chạy đầu-cuối trên dữ liệu thật, không có error output.

### Deliverables Sprint 1

1. Data dictionary và data contract.
2. Báo cáo EDA/validation: node, cạnh, feature, nhãn, split, degree, edge type và timestamp.
3. Thiết kế và hiện thực pipeline tải/kiểm định/tiền xử lý/graph conversion sau khi bước triển khai được phê duyệt.
4. Cấu hình sampling hoặc subgraph phù hợp tài nguyên, kèm benchmark RAM/VRAM/thời gian.
5. Output validation, assert và báo cáo kiểm định trong notebook đã chạy.
6. Manifest của dữ liệu đã xử lý và tài liệu quyết định kỹ thuật.

## 9. Kế hoạch công việc Sprint 2

Đề xuất thời lượng: **2 tuần (10 ngày làm việc)**, bắt đầu khi Sprint 1 đạt checklist nghiệm thu.

| Mốc | Thời gian dự kiến | Công việc | Tiêu chí hoàn thành |
|---|---:|---|---|
| S2-M1: Protocol chung | Ngày 1 | Chốt metric, seed, early stopping, checkpoint, loss, sampling và ngân sách | `experiment_protocol` được phê duyệt |
| S2-M2: GCN baseline | Ngày 2–3 | Định nghĩa GCN trong notebook; chạy quick forward/train/validation | Loss hữu hạn, metric được tính đúng, checkpoint được lưu đủ metadata |
| S2-M3: GraphSAGE baseline | Ngày 4–5 | Xây dựng GraphSAGE trên cùng interface và dữ liệu | Chạy qua cùng experiment runner |
| S2-M4: Thực nghiệm kiểm soát | Ngày 6–8 | Chạy cấu hình chốt, theo dõi tài nguyên, tối thiểu 3 seed nếu khả thi | Có run hoàn chỉnh, không dùng test để tuning |
| S2-M5: Test & so sánh | Ngày 9 | Khóa cấu hình, test checkpoint tốt nhất, tổng hợp AUC/AP | Bảng kết quả có trung bình và độ phân tán nếu nhiều seed |
| S2-M6: Bàn giao | Ngày 10 | Phân tích kiến trúc, sai số và hạn chế; đóng gói artifact | Checklist nghiệm thu Sprint 2 đạt |

### Ma trận điều kiện so sánh công bằng

| Thành phần | Phải giống nhau | Được khác nhưng phải ghi lại |
|---|---|---|
| Dữ liệu | Dataset fingerprint, split, feature pipeline, chính sách lớp nền | Không |
| Sampling | Seed, node đích, nguyên tắc tạo batch | Fan-out/số lớp nếu kiến trúc bắt buộc |
| Huấn luyện | Optimizer, learning rate, batch size, epoch tối đa, patience, loss policy | Hidden size/dropout chỉ khi có lý do và có thí nghiệm kiểm soát |
| Đánh giá | Positive class = 1, ROC-AUC, AP, checkpoint rule | Threshold metric chỉ là bổ trợ |
| Ngân sách | Số seed và giới hạn tài nguyên tương đương | Thời gian thực tế được báo cáo như kết quả |

### Checklist nghiệm thu Sprint 2

- [ ] GCN và GraphSAGE dùng chung interface và experiment protocol.
- [ ] Positive class và cách tính AUC/AP được xác nhận bằng validation cell và output.
- [ ] Validation quyết định checkpoint; test chỉ được dùng sau khi khóa cấu hình.
- [ ] Có baseline xử lý mất cân bằng được chốt bằng validation.
- [ ] Mỗi run lưu cấu hình, seed, phiên bản dữ liệu, log và checkpoint.
- [ ] Checkpoint có thể tải lại và tái tạo metric trong sai số cho phép.
- [ ] Kết quả có AUC, AP, thời gian, RAM/VRAM và độ phân tán giữa seed nếu khả thi.
- [ ] Báo cáo nêu rõ thiết lập static/transductive và các hạn chế đối với dynamic fraud-ring detection.

### Deliverables Sprint 2

1. Experiment protocol dùng chung cho GCN và GraphSAGE.
2. Hai baseline GCN/GraphSAGE và pipeline train-validation-inference sau khi triển khai được phê duyệt.
3. Checkpoint tốt nhất của từng mô hình kèm cấu hình và fingerprint dữ liệu.
4. Log, learning curve, prediction/metric artifact cần thiết để tái kiểm tra.
5. Bảng so sánh ROC-AUC, AP, độ ổn định, thời gian và tài nguyên.
6. Báo cáo phân tích ưu/nhược điểm của hai kiến trúc và giới hạn của baseline tĩnh.

## 10. Rủi ro và hướng xử lý

| Rủi ro | Khả năng/tác động | Dấu hiệu | Hướng xử lý |
|---|---|---|---|
| Dataset lớn gây hết RAM/VRAM | Cao/Cao | Loader bị kill, CUDA OOM, swap tăng mạnh | Đọc có kiểm soát, giảm precision khi an toàn, neighbor sampling, batch nhỏ, gradient accumulation; ghi benchmark trước full run |
| Mất cân bằng lớp nghiêm trọng | Cao/Cao | AUC khá nhưng AP thấp; mô hình dự đoán phần lớn lớp 0 | Theo dõi AP là metric chính, class weight/sampling chỉ fit từ train, báo cáo PR curve và metric theo lớp |
| Rò rỉ dữ liệu qua preprocessing | Trung bình/Cao | Metric validation/test cao bất thường | Tách rõ transform mặc định và ablation transductive; validation split; lưu provenance từng phép biến đổi |
| Rò rỉ cấu trúc trong thiết lập transductive bị hiểu sai | Trung bình/Cao | Kết quả không tái hiện trong inductive deployment | Công bố rõ transductive; tách thí nghiệm inductive/temporal về sau nếu cần |
| Split ngẫu nhiên không phản ánh diễn biến thời gian | Cao/Cao đối với mục tiêu “dynamic” | Kết quả giảm khi đánh giá theo thời gian | Giữ split gốc cho baseline, ghi hạn chế; đề xuất temporal split ở sprint temporal nhưng không tự thay đổi Sprint 1–2 |
| Lớp 2/3 bị coi nhầm là normal | Trung bình/Cao | Loss/metric bị pha loãng hoặc sai định nghĩa | Chỉ tính loss/metric lớp 0/1; lớp nền chỉ tham gia cấu trúc theo A4 |
| Hướng cạnh/self-loop không rõ | Trung bình/Trung bình | Kết quả thay đổi lớn giữa biến thể | Giữ canonical graph, cấu hình hóa biến đổi, chạy ablation nhỏ và ghi quyết định |
| Neighbor sampling làm metric dao động | Trung bình/Trung bình | Chênh lệch lớn giữa seed/batch | Cố định seed, đánh giá coverage, nhiều seed, inference ổn định theo quy trình chốt |
| So sánh GCN và GraphSAGE thiếu công bằng | Trung bình/Cao | Một mô hình có ngân sách/tuning nhiều hơn | Dùng ma trận điều kiện chung, ghi mọi ngoại lệ, giới hạn số thử nghiệm tương đương |
| Metric sai positive class hoặc sai mask | Trung bình/Cao | AP bất thường, số mẫu đánh giá không khớp | Validation cell xác nhận positive class = 1 và số node mỗi split trước khi tính metric |
| Không tái lập được kết quả | Trung bình/Cao | Chạy lại sai khác lớn hoặc thiếu cấu hình | Seed, deterministic mode khi khả thi, version lock, data fingerprint, run manifest |
| Tệp checkpoint/log chiếm nhiều dung lượng | Trung bình/Trung bình | Ổ đĩa đầy trong nhiều seed | Retention policy: giữ best/last, nén metric, dọn artifact theo run manifest sau khi được phê duyệt |
| Baseline node-level chưa chứng minh fraud-ring | Cao/Cao về diễn giải | Có dự đoán node nhưng không có ring output | Giới hạn tuyên bố; định nghĩa ring-level task/evaluation ở giai đoạn sau với phê duyệt riêng |

## 11. Cổng phê duyệt giữa hai sprint

Chỉ bắt đầu Sprint 2 khi các điều kiện sau được đáp ứng:

- Sprint 1 vượt qua toàn bộ kiểm tra schema và toàn vẹn dữ liệu mức blocker.
- Chính sách lớp nền, hướng cạnh, self-loop, preprocessing và sampling đã được phê duyệt.
- Pipeline chạy đầu-cuối trên máy mục tiêu trong giới hạn tài nguyên.
- Data fingerprint, split metadata và experiment protocol bản nháp đã được lưu.
- Không còn điểm chưa rõ có thể làm thay đổi định nghĩa bài toán hoặc cách tính metric.

Sau khi tài liệu kế hoạch này được phê duyệt, bước tiếp theo mới là bắt đầu Sprint 1; chưa thực hiện bất kỳ hoạt động triển khai nào trong tài liệu hiện tại.
