# Kế hoạch Sprint 4 — Community Detection để bổ trợ Fraud Detection

## 1. Bối cảnh và mục tiêu

Sprint 3 đã chọn TGAT là pipeline có validation AP cao nhất. Output hiện tại vẫn là xác
suất fraud ở cấp node. Sprint 4 tập trung vào **Community Detection** để trả lời ba câu hỏi:

1. Leiden tìm được cấu trúc community có chất lượng và quy mô hợp lý hay không?
2. Fraud có tập trung bất thường trong một số community hay không?
3. Feature/risk score ở cấp community có cải thiện TGAT so với baseline Sprint 3 hay
   không?

Kết quả mong đợi là một pipeline phát hiện **risky community** có thể kiểm chứng bằng
EDA, hình trực quan và benchmark định lượng. Chỉ gọi một community là risky community
khi nó thỏa các tiêu chí cấu trúc và rủi ro đã định nghĩa; không xem mọi community là
một nhóm gian lận có phối hợp.

## 2. Phạm vi và nguyên tắc thực nghiệm

- Giữ nguyên dataset fingerprint, feature `zero_indicator` 34D, split và ba seed
  42/43/44 của protocol hiện tại.
- Dùng graph undirected `structural_coalesced` làm graph chính cho Community Detection.
  Đây là graph đã thêm cạnh ngược và gộp các cạnh trùng theo cặp node trong Sprint 3.
- Community Detection chỉ dùng cấu trúc kết nối giữa các node, tuyệt đối không nhận
  nhãn `y`, thông tin split hoặc xác suất dự đoán của model làm đầu vào.
- Thiết lập vẫn là full-history transductive node classification. Việc dùng toàn bộ
  cấu trúc kết nối của graph là phù hợp với thiết lập này nhưng không được diễn giải
  thành dự đoán fraud tương lai.
- Nhãn train dùng để tạo community-risk feature, validation dùng để chọn cấu hình;
  test chỉ được mở một lần sau khi khóa thuật toán, resolution, feature và quy tắc
  chọn checkpoint.
- Average Precision (AP) là metric chính vì fraud là lớp rất hiếm; ROC-AUC là metric
  phụ. Không kết luận hiệu quả chỉ từ modularity hoặc một hình graph đẹp.

Sprint 1 đã kiểm tra số node/cạnh, miền node ID, cạnh tự nối (self-loop), node cô lập,
bậc node (degree), edge type, timestamp và fingerprint của dữ liệu gốc. Sprint 4 không
lặp lại phần EDA đó.
Khi nạp dữ liệu chỉ đối chiếu nhanh dataset SHA-256 và số cạnh `7.994.520` của
`structural_coalesced` với artifact Sprint 3 để tránh chạy nhầm dữ liệu.

## 3. Phương pháp Community Detection

### 3.1. Thuật toán

- Phương pháp chính và duy nhất trong phạm vi bắt buộc: **Leiden** trên graph
  `structural_coalesced`.
- Dùng modularity làm hàm mục tiêu, resolution `1.0` và seed `42` làm cấu hình
  cố định ban đầu. Không thay resolution sau khi xem test.
- Chạy thử trên một graph con có seed cố định để đo thời gian và mức RAM cao nhất
  trước khi chạy toàn bộ graph 3,7 triệu node. Thư viện Leiden chưa có trong
  `requirements.txt`; chỉ thêm dependency sau khi lần chạy thử xác nhận tương thích
  Python 3.12 và giới hạn tài nguyên.

Leiden được ưu tiên vì có bước tinh chỉnh (refinement phase), giúp tạo community liên
thông tốt hơn Louvain. Tuy nhiên, modularity có giới hạn về độ phân giải (resolution
limit): trên graph lớn, nó có thể gộp các nhóm nhỏ đáng lẽ nên tách riêng. Vì vậy,
modularity chỉ mô tả chất lượng cách chia nhóm, không phải bằng chứng độc lập rằng
community hữu ích cho fraud.

Mỗi lần chạy phải lưu thuật toán, phiên bản thư viện, seed, resolution, số node/edge,
thời gian chạy, mức RAM cao nhất, số community và SHA-256 của dataset.

### 3.2. Feature cấp community

Với mỗi node, tạo hai nhóm feature để ablation độc lập:

- **Structural, không dùng nhãn:** logarit của kích thước community
  (`log_community_size`) để giảm chênh lệch giữa nhóm rất lớn và rất nhỏ; số cạnh và
  mật độ cạnh nội bộ; tỷ lệ cạnh của node nối tới node cùng community
  (`internal_degree / total_degree`); và conductance của community.
- **Community risk, chỉ dùng nhãn train:** fraud rate của community. Đặt trước
  ngưỡng `MIN_LABELED_COMMUNITY=20`; community không đủ 20 train node có nhãn nhận
  fraud rate chung của toàn bộ train thay vì một tỷ lệ 0%/100% thiếu tin cậy.

Community ID là mã tùy ý nên không đưa trực tiếp vào model như một biến số liên tục.
Điểm rủi ro của train node phải dùng leave-one-out để nhãn của chính node không tự rò
vào feature. Điểm rủi ro của validation/test chỉ được tính từ nhãn train. Bayesian
smoothing có thể được thêm ở phần mở rộng, nhưng ngưỡng tối thiểu và leave-one-out là
bắt buộc.

## 4. Các bước EDA

Phần này chỉ EDA kết quả Community Detection. EDA graph gốc không làm lại vì đã hoàn
thành ở Sprint 1.

### Bước 1 — Khảo sát quy mô và chất lượng community

- Thống kê số community; kích thước nhỏ nhất, trung vị, p90/p99 (90%/99% community có
  kích thước không vượt quá giá trị này), lớn nhất; và tỷ lệ community chỉ có một node.
- Đo modularity, coverage, conductance và internal-edge ratio.
- Kiểm tra Leiden có tạo một community quá lớn chứa phần lớn graph hoặc quá nhiều
  community rất nhỏ hay không. Hai trường hợp này làm feature cấp community kém hữu ích.
- Không dùng riêng modularity để kết luận Community Detection cải thiện fraud
  detection; kết luận cuối phải dựa vào fraud lift trên validation và ablation model.

### Bước 2 — Kiểm tra mức tập trung fraud, không dùng test để khám phá

- Với từng community, báo số node có nhãn (`labeled_count`), số node fraud
  (`fraud_count`), tỷ lệ fraud (`fraud_rate`) và
  `fraud_lift = community_fraud_rate / global_fraud_rate`.
- Chỉ dùng train fraud rate để xếp hạng và chọn risky community. Dùng validation fraud
  rate/lift để kiểm tra thứ hạng này có còn đúng trên dữ liệu không dùng để tính risk
  feature hay không.
- Không dùng test fraud count/rate/lift trong EDA hoặc chọn community.
- Phân tích thêm theo community size, conductance và internal-edge ratio để tránh hiểu
  nhầm một community rủi ro chỉ vì nó lớn hoặc có nhiều node được gán nhãn.

### Bước 3 — Chọn risky community để minh họa

Một community chỉ được đưa vào danh sách điều tra khi đồng thời có:

- quy mô tối thiểu và đủ node có nhãn để điểm rủi ro đáng tin cậy;
- train fraud lift cao theo quy tắc được xác định trước khi xem validation/test;
- liên kết nội bộ tốt, thể hiện qua conductance thấp hoặc internal-edge ratio
  cao;
- community không trở thành risky chỉ vì một fraud node duy nhất.

Kết quả là bảng ứng viên gồm community ID, kích thước, số cạnh nội bộ, conductance, số
node train/validation được gán nhãn, fraud rate/lift và các node có xác suất fraud cao
nhất theo model. Đây là danh sách **risky community**, chưa phải bằng chứng nghiệp vụ
về một đường dây gian lận.

## 5. Kế hoạch trực quan hóa

| Hình bắt buộc | Nội dung | Bằng chứng cần thể hiện |
|---|---|---|
| 01 | Histogram và phân bố tích lũy ngược (CCDF) của kích thước community, dùng trục log | Cách chia nhóm không bị chi phối hoàn toàn bởi community một node hoặc một community quá lớn |
| 02 | Bar/scatter fraud count, fraud rate và fraud lift theo community | Fraud có tập trung vào một số community và không chỉ phản ánh community size |
| 03 | Graph con chỉ chứa node/cạnh của 3–5 risky communities, được chọn bằng quy tắc train | Minh họa cấu trúc kết nối; đây là bằng chứng định tính, không thay thế metric |
| 04 | Đường Precision–Recall (PR curve) của TGAT và TGAT + community | Ở nhiều ngưỡng dự đoán, mô hình mới có giữ precision tốt hơn khi recall tăng hay không |
| 05 | Biểu đồ AP và ROC-AUC trung bình của ablation A/B/C/D | Xác định phần đóng góp của structural feature và train-only risk feature |

Quy tắc vẽ graph con:

- Chỉ chọn community theo quy tắc đã xác định trước, không chọn vì hình “đẹp”.
- Nếu community quá lớn để vẽ, lấy một phần theo quy tắc cố định như các node có nhiều
  cạnh nhất (top-degree) hoặc phần lõi liên kết chặt (k-core); ghi rõ số node bị lược bỏ.
- Phân biệt fraud, normal, background và unlabeled bằng màu/ký hiệu riêng; không coi
  node background là normal.
- Dùng cùng thang màu, seed tạo bố cục và chú thích giữa các hình để so sánh được.

## 6. Tích hợp vào model và ablation

Giữ nguyên TGAT Sprint 3 làm mốc so sánh và chạy bốn cấu hình:

| Cấu hình | Đặc trưng bổ sung | Mục đích |
|---|---|---|
| A | Không có | Tái lập TGAT gốc |
| B | Structural community features | Đo giá trị của cấu trúc community không dùng nhãn |
| C | Train-only community-risk feature | Đo giá trị của tín hiệu rủi ro có ngưỡng tối thiểu và leave-one-out |
| D | Structural + community-risk features | Đánh giá cấu hình community đầy đủ |

Mọi cấu hình dùng cùng split, seed, bộ tối ưu, số vòng huấn luyện (epoch) tối đa và quy
tắc dừng sớm; checkpoint được chọn bằng validation AP. Nếu chuẩn hóa feature thì các
tham số chuẩn hóa chỉ được tính trên train. Báo AP là metric chính, ROC-AUC là metric
phụ. Với ba
seed, báo trung bình ± độ lệch chuẩn và mức thay đổi (`delta AP`, `delta ROC-AUC`) của
từng cấu hình so với A.

Chọn duy nhất cấu hình thắng bằng AP validation trung bình. Sau khi khóa lựa chọn, chỉ so
test của A với cấu hình thắng; không dùng test để xếp hạng lại B/C/D.

## 7. Tiêu chí kết luận Community Detection có hiệu quả

Community Detection chỉ được kết luận là hữu ích cho bài toán khi có bằng chứng ở cả
ba lớp:

1. **Cấu trúc:** Leiden tạo phân hoạch hợp lệ; modularity, size distribution,
   conductance và internal-edge ratio được báo cáo đầy đủ.
2. **Tập trung rủi ro:** community được xếp hạng chỉ bằng train risk vẫn có validation
   fraud lift lớn hơn 1; không chọn community bằng validation/test label.
3. **Giá trị dự đoán:** ít nhất một cấu hình B/C/D có AP validation trung bình cao hơn A,
   cải thiện xuất hiện ở ít nhất 2/3 seed; cấu hình đã khóa sau đó có test AP cao hơn A.

Nếu chỉ đạt (1), kết luận là graph có community structure. Nếu đạt (1) và (2) nhưng
không đạt (3), Community Detection hữu ích cho khoanh vùng/giải thích nhưng chưa chứng
minh cải thiện node classifier. ROC-AUC được dùng để bổ sung diễn giải, không được dùng
thay AP để tuyên bố thành công. Nếu không đạt các tiêu chí, vẫn báo cáo kết quả âm và
không điều chỉnh threshold bằng test để tạo kết luận thuận lợi.

## 8. Thứ tự triển khai đề xuất

### Giai đoạn 1 — Chuẩn bị và kiểm tra tính khả thi

- Tái sử dụng quy ước dữ liệu và graph đã được kiểm tra ở Sprint 1–3; chỉ đối chiếu
  fingerprint và số cạnh, không làm lại EDA graph.
- Chạy lại TGAT gốc với seed 42 để xác nhận kỹ thuật và metric của Sprint 3 có thể tái
  lập. Đây là run kiểm tra tạm thời; xóa raw checkpoint/run sau khi đã lưu metric kiểm
  chứng vào artifact Sprint 4.
- Chạy thử Leiden trên graph con, chốt thư viện và giới hạn RAM/thời gian.
- Định nghĩa trước cấu trúc các file kết quả cần lưu.

### Giai đoạn 2 — Community EDA

- Chạy Leiden với cấu hình cố định trên toàn bộ graph.
- Tạo bảng kích thước community, modularity, conductance và internal-edge ratio.
- Tạo hình 01 và khóa community assignment trước khi phân tích nhãn.

### Giai đoạn 3 — Phân tích mức tập trung fraud và trực quan hóa

- Tính điểm rủi ro chỉ từ nhãn train, đánh giá trên validation.
- Tạo hình 02–03 và bảng ứng viên risky community.

### Giai đoạn 4 — Tích hợp community feature vào model

- Tạo community feature có kiểm soát label leakage.
- Chạy ablation A/B/C/D trên seed 42/43/44.
- Khóa cấu hình bằng validation AP, sau đó mới chạy test và tạo hình 04–05.

### Giai đoạn 5 — Tổng hợp

- Viết `docs/sprint4_report.md`, nêu cả kết quả ủng hộ và kết quả âm.
- Cập nhật protocol, data/artifact provenance và README.
- Kiểm tra notebook không có error output và cờ official mặc định là `False`.

## 9. Công việc mở rộng — chỉ làm khi còn thời gian

Thực hiện theo thứ tự ưu tiên sau, không phải điều kiện bắt buộc để hoàn thành Sprint 4:

1. So sánh Leiden với Louvain.
2. Chạy nhiều resolution và dùng ARI/NMI để đo mức giống nhau giữa các cách chia nhóm
   qua nhiều seed.
3. So với random partition, tức cách chia ngẫu nhiên nhưng giữ nguyên phân bố kích
   thước community.
4. Thêm Bayesian smoothing để giảm score cực đoan của community nhỏ và confidence
   interval để thể hiện khoảng bất định của fraud rate/lift.
5. Chạy degree-stratified permutation test: xáo nhãn trong các nhóm node có degree gần
   nhau để kiểm tra fraud lift có vượt kết quả ngẫu nhiên hay không.
6. Vẽ recall–investigation budget curve: tỷ lệ fraud tìm được khi chỉ có nguồn lực
   kiểm tra một tỷ lệ node nhất định.
7. Thêm logistic regression/MLP baseline chỉ dùng community features để có một mô hình
   đơn giản làm đối chứng.

Các phân tích mở rộng chỉ được dùng để tăng độ mạnh của bằng chứng. Không được xem test
để quyết định có đưa một biến thể vào báo cáo chính hay không.

## 10. Kết quả cần bàn giao và cấu trúc artifact

- Notebook/code có thể chạy lại Community Detection, EDA, quá trình tạo feature và
  ablation.
- `artifacts/metrics/sprint4_community_results.json`: config, structural metrics,
  fraud count/rate/lift, model metrics và kết quả mở rộng nếu có chạy.
- `artifacts/metrics/sprint4_community_assignments.*`: bảng ánh xạ
  `node_id → community_id` kèm graph/dataset fingerprint; chọn định dạng nén phù hợp
  vì graph lớn.
- `artifacts/figures/sprint4/01_...png` đến `05_...png` theo bảng trực quan hóa bắt
  buộc; hình mở rộng đánh số tiếp theo.
- `docs/sprint4_report.md`: protocol, kết quả, giới hạn và kết luận.
- Bảng ứng viên risky community không chứa dữ liệu định danh ngoài node ID đã ẩn danh.

## 11. Điều kiện hoàn thành

- Mọi node đều có community ID và có manifest ghi đủ thông tin để tái lập kết quả.
- Có community-size EDA, modularity, fraud count/rate/lift và đầy đủ hình 01–05.
- Không có label leakage trong detector hoặc community-risk feature.
- Ablation A/B/C/D chạy đủ ba seed và đánh giá toàn bộ validation split; test chỉ so A
  với cấu hình thắng đã chọn bằng validation AP.
- Model selection chỉ dùng validation AP; test không dùng để chọn resolution, feature,
  threshold hoặc checkpoint.
- Báo cáo phân biệt rõ node classification và risky-community discovery; không đưa ra
  tuyên bố vượt quá bằng chứng từ DGraphFin.

## 12. Tài liệu phương pháp

- Traag, Waltman và van Eck (2019), [*From Louvain to Leiden: guaranteeing
  well-connected communities*](https://doi.org/10.1038/s41598-019-41695-z),
  Scientific Reports.
- Fortunato và Barthélemy (2007), [*Resolution limit in community
  detection*](https://doi.org/10.1073/pnas.0605965104), PNAS.
- Huang et al. (2022), [*DGraph: A Large-Scale Financial Dataset for Graph Anomaly
  Detection*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract-Datasets_and_Benchmarks.html),
  NeurIPS Datasets and Benchmarks.
