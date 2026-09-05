# Kế hoạch Sprint 5.2 — Phân tích dự đoán sai và kiểm chứng tín hiệu TGAT

## 1. Bối cảnh

Sprint 5 đã giải thích 80 dự đoán của **TGAT + community-risk** bằng
GNNExplainer và kiểm tra lại bằng các phép thay đổi trực tiếp. Kết quả đáng tin nhất
nằm ở cấp nhóm feature, nhưng Sprint 5 chưa khóa một ngưỡng phân loại để xác định
false positive (FP) và false negative (FN), đồng thời chưa đo trực tiếp TGAT phụ thuộc
vào thời gian và thông tin hàng xóm đến mức nào.

Vì vậy Sprint 5.2 tập trung vào hai hướng ưu tiên:

1. giải thích có hệ thống các dự đoán sai;
2. thay đổi riêng timestamp, hàng xóm và message passing để kiểm tra TGAT thực sự nhạy
   với thông tin nào.

Sprint 5.2 vẫn là nghiên cứu giải thích một model đã huấn luyện. Sprint này không
khẳng định feature được chọn là nguyên nhân nghiệp vụ gây fraud và không gọi
neighborhood được quan sát là fraud ring đã xác nhận.

## 2. Câu hỏi nghiên cứu

### 2.1. Vì sao model dự đoán sai?

- Sau khi khóa threshold trên validation, model tạo bao nhiêu FP và FN trên từng
  split?
- Ở FP, feature hoặc community-risk nào đang đẩy fraud logit lên?
- Ở FN, đầu vào nào đang đẩy fraud logit xuống hoặc không cung cấp đủ tín hiệu fraud?
- Các mẫu đóng góp ở FP/FN khác gì so với TP/TN có cùng phía dự đoán và score gần
  tương đương?

### 2.2. TGAT đã học gì từ graph và thời gian?

- Dự đoán thay đổi bao nhiêu khi xóa chênh lệch thời gian giữa các event?
- Dự đoán thay đổi bao nhiêu khi phá liên kết giữa event và nội dung hàng xóm?
- Dự đoán còn giữ được bao nhiêu khi tắt toàn bộ thông điệp từ hàng xóm nhưng giữ
  feature của target và community-risk?

## 3. Kết quả cần bàn giao

Sprint 5.2 cần tạo được ba kết quả chính:

1. một threshold được chọn và khóa chỉ từ validation, kèm confusion matrix và danh
   sách TP/FP/FN/TN;
2. phân tích trực tiếp các feature và community-risk đang hỗ trợ phía dự đoán đúng
   hoặc sai;
3. bảng ablation định lượng vai trò của thời gian, hàng xóm và message passing.

Kết quả khoa học không bắt buộc phải cho thấy mọi thành phần của TGAT đều hữu ích. Một
kết quả âm nhưng có protocol khóa trước, artifact đầy đủ và diễn giải đúng giới hạn
vẫn được xem là hoàn thành sprint.

## 4. Phạm vi và đầu vào bị khóa

- Dataset: `data/dgraphfin.npz`, SHA-256
  `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85`.
- Model chính: TGAT + community-risk, variant C của Sprint 4.
- Checkpoint chính: seed 42, SHA-256
  `558db58e8c788edb8f5862b64093ef5a1a463debadd80e7e119d230ca602b61e`.
- Thiết lập graph: `temporal_event_mirror`, full-history transductive classification.
- Neighborhood: một hop, tối đa 15 event, materialize bằng seed cố định trước mọi
  phép giải thích/can thiệp.
- Trọng số model luôn ở chế độ `eval` và bị đóng băng.
- Community-risk tiếp tục chỉ đi vào classifier của target sau message passing; không
  truyền nhãn train qua hàng xóm.
- Mọi cấu hình, target ID, seed và chỉ số chính phải được ghi vào protocol lock trước
  khi tổng hợp kết quả test.

Test đã được mở trong Sprint 4 và Sprint 5. Vì thế kết quả test của Sprint 5.2 được
dùng để kiểm tra khả năng lặp lại theo protocol mới, nhưng không được mô tả như một
blind holdout hoàn toàn mới.

## 5. Giai đoạn 0 — Khóa protocol và kiểm tra tái lập

Trước khi phân tích FP/FN:

1. kiểm tra hash dataset, checkpoint, community feature và các artifact Sprint 5;
2. tái tạo prediction của checkpoint seed 42 trên toàn bộ validation và test;
3. kiểm tra adapter/counterfactual runner cho logit giống forward gốc trong điều kiện
   không can thiệp, với sai số tuyệt đối tối đa `1e-6`;
4. materialize và lưu local batch của các target sẽ giải thích;
5. khóa threshold rule, target-selection rule, perturbation definitions, seed và
   metric trong `sprint5_2_protocol_lock.json`;
6. chỉ sau khi lock hợp lệ mới tổng hợp kết quả test.

Nếu một thay đổi code làm khác prediction gốc, phải sửa và tạo lại lock; không được
tiếp tục dùng kết quả từ hai implementation không tương đương.

## 6. Giai đoạn 1 — Khóa threshold và tạo cohort lỗi

### 6.1. Chọn threshold

Vì dự án chưa có chi phí nghiệp vụ cụ thể cho FP và FN, ngưỡng nghiên cứu mặc định
được chọn bằng cách **tối đa F1 của lớp fraud trên toàn bộ validation**. Nếu nhiều
ngưỡng có cùng F1, ưu tiên recall cao hơn, sau đó chọn ngưỡng nhỏ hơn. Quy tắc và
threshold số học cuối cùng phải được lưu trước khi áp dụng lên test.

Ngoài threshold chính, báo đường precision–recall và confusion matrix theo threshold
để người đọc thấy đánh đổi. Các ngưỡng phụ chỉ là sensitivity analysis, không được
dùng để chọn lại kết luận sau khi xem test.

### 6.2. Tạo TP/FP/FN/TN

Áp dụng cùng threshold đã khóa cho validation và test:

| Cohort | Điều kiện |
|---|---|
| TP | label fraud, score lớn hơn hoặc bằng threshold |
| FP | label normal, score lớn hơn hoặc bằng threshold |
| FN | label fraud, score nhỏ hơn threshold |
| TN | label normal, score nhỏ hơn threshold |

Trên toàn split, báo ít nhất: số lượng, precision, recall, F1, predicted-positive
rate và confusion matrix. AP và ROC-AUC vẫn được giữ để đối chiếu nhưng không dùng
thay cho phân tích theo threshold.

### 6.3. Chọn target để giải thích sâu

Thống kê lỗi dùng toàn bộ node. Phần giải thích sâu lấy tối đa 20 node cho mỗi cohort
trên mỗi split:

- với FP/FN: một nửa là lỗi gần threshold nhất, một nửa là lỗi tự tin nhất;
- với mỗi FP, chọn một TP control ở cùng phía dự đoán; với mỗi FN, chọn một TN control
  ở cùng phía dự đoán;
- ghép control theo fraud-score bin, `log1p(degree)` và số sampled event; nếu có nhiều
  ứng viên bằng nhau thì chọn node ID nhỏ hơn;
- nếu cohort có dưới 20 node thì dùng toàn bộ và ghi rõ cỡ mẫu thực tế;
- không loại node vì lời giải thích khó đọc hoặc vì neighborhood chỉ có một event.

Target selection chỉ đọc label, prediction, degree và neighborhood metadata; tuyệt
đối không đọc explainer mask hay kết quả perturbation trước khi khóa danh sách.

## 7. Giai đoạn 2 — Giải thích trực tiếp FP và FN

### 7.1. Feature và community-risk

Với mỗi target đã khóa:

1. đưa lần lượt từng feature về baseline train đã dùng ở Sprint 5;
2. chạy lại frozen model và lưu fraud logit/score trước và sau;
3. chạy GNNExplainer feature-only hoặc tái sử dụng artifact Sprint 5 nếu target,
   checkpoint và frozen batch trùng hoàn toàn;
4. so GNNExplainer với direct ablation bằng Spearman và top-5 Jaccard;
5. báo riêng rank và tác động của community-risk.

Gọi `z` là fraud logit và `z(-i)` là logit sau khi bỏ/đưa đầu vào `i` về baseline.
Định nghĩa tác động đẩy về phía dự đoán sai:

- FP: `error_push(i) = z - z(-i)`;
- FN: `error_push(i) = z(-i) - z`.

`error_push > 0` nghĩa là đầu vào ban đầu đang hỗ trợ phía dự đoán sai. Với TP/TN,
dùng công thức tương tự theo phía lớp được dự đoán để tạo nhóm control. Báo cả giá trị
có dấu và trị tuyệt đối; không chỉ đếm top-k mask.

### 7.2. Tổng hợp lỗi

Báo riêng validation/test và FP/FN:

- feature xuất hiện nhiều nhất trong direct top-5 và GNNExplainer top-5;
- median, IQR và tỷ lệ `error_push > 0` của từng feature;
- tác động community-risk và tỷ lệ nó nằm trong direct/GNNExplainer top-5;
- tổng tác động feature so với tổng tác động graph;
- event recency, degree và số event quanh target;
- so sánh với TP/TN control đã ghép.

Kết luận phải dùng cách nói “đầu vào hỗ trợ dự đoán sai của model”. Không chuyển thành
khẳng định rằng đầu vào đó gây ra fraud hoặc gây ra lỗi trong thế giới thực.

## 8. Giai đoạn 3 — Ablation TGAT ở thời điểm inference

Các ablation dưới đây dùng cùng frozen model và cùng materialized neighborhood. Đây
là **inference-time counterfactual**, không phải train lại một kiến trúc mới.

| ID | Can thiệp | Phần được giữ nguyên | Câu hỏi |
|---|---|---|---|
| FULL | Không can thiệp | Toàn bộ đầu vào | Mốc so sánh |
| T-CONST | Đặt mọi `edge_delta` trong một neighborhood về median của chính neighborhood đó | Node, edge và multiset hàng xóm | Model có dùng khác biệt thời gian tương đối không? |
| N-BASE | Đưa feature của các neighbor về baseline train, giữ feature target | Edge, timestamp và số hàng xóm | Nội dung hàng xóm có đóng góp không? |
| MP-OFF | Bỏ toàn bộ neighbor event/message, giữ root/target path và community-risk | Feature target và classifier | Tổng đóng góp của message passing là bao nhiêu? |

Yêu cầu kỹ thuật:

- `T-CONST` chỉ thay `edge_delta` đầu vào time encoder;
- `N-BASE` không được thay feature của target;
- `MP-OFF` phải có unit test chứng minh target/root path và community-risk vẫn đi qua
  classifier, chỉ aggregated neighbor message bị loại;
- kiểm tra shape, finite logit và ID alignment sau mỗi can thiệp.

Chạy ablation trên toàn bộ validation và test. Với mỗi cấu hình, báo AP và ROC-AUC cùng
mức thay đổi so với `FULL`. Đây là hai metric đánh giá khả năng xếp hạng đã được dùng
xuyên suốt các sprint. Kết quả chỉ được diễn giải là độ nhạy của frozen model, không
phải tác động của đầu vào trong thế giới thực.

## 9. Work breakdown

| Gói việc | Ưu tiên | Phụ thuộc | Đầu ra |
|---|---:|---|---|
| W0. Reproduce prediction và khóa protocol | P0 | Artifact Sprint 4–5 | Protocol lock, provenance, equivalence test |
| W1. Chọn threshold và tạo TP/FP/FN/TN | P0 | W0 | Prediction table, threshold curve, cohort IDs |
| W2. Chọn/match target và chạy direct explanation | P0 | W1 | Feature/community-risk error attribution |
| W3. Chạy time/neighbor/message-passing ablation | P0 | W0–W1 | Full-split ablation metrics và paired CI |
| W4. Dựng hình, notebook, report và manifest | P0 | W1–W3 | Notebook 07, report Sprint 5.2, artifact gate |
| W5. Lặp robustness trên checkpoint seed 43/44 | P1 | W3 | Cross-checkpoint sensitivity appendix |

Thứ tự triển khai là `W0 → W1 → (W2 và W3) → W4`; W5 chỉ bắt đầu sau khi toàn bộ
P0 đã đạt gate.

## 10. Artifact dự kiến

- `artifacts/metrics/sprint5_2_protocol_lock.json`: hash, threshold rule, target rule,
  perturbation, seed và metric đã khóa.
- `artifacts/metrics/sprint5_2_predictions.csv.gz`: prediction và TP/FP/FN/TN cho toàn
  validation/test.
- `artifacts/metrics/sprint5_2_targets.csv`: target giải thích sâu và control matching.
- `artifacts/metrics/sprint5_2_error_attribution.json`: feature và community-risk
  attribution cho từng target.
- `artifacts/metrics/sprint5_2_ablation_results.json`: full-split metric, node-level
  delta và bootstrap interval cho từng ablation.
- `artifacts/metrics/sprint5_2_manifest.json`: provenance, checks và artifact hashes.
- `notebooks/07_tgat_error_analysis.ipynb`: notebook trình bày đã execute.
- `docs/sprint5.2_report.md`: báo cáo kết quả sau khi thí nghiệm hoàn tất.

Các mảng lớn có thể lưu ở NPZ thay vì nhúng vào JSON, nhưng manifest phải ghi path,
SHA-256, shape, dtype và khóa nối với bảng target.

## 11. Bộ hình dự kiến

| File | Nội dung |
|---|---|
| `01_threshold_and_errors.png` | Precision–recall, threshold khóa và confusion matrix |
| `02_fp_fn_feature_push.png` | Feature/community-risk đẩy FP và FN so với control |
| `03_tgat_input_ablation.png` | Delta metric và logit của T/N/MP ablation |
| `04_error_case_panels.png` | Một số case FP/FN có provenance đầy đủ, không chọn theo hình đẹp |

Hình case study phải được chọn bằng quy tắc khóa trước, ví dụ FP/FN có
`absolute error_push` lớn nhất trong từng stratum; không chọn thủ công sau khi xem
kết quả.

## 12. Gate hoàn thành

Sprint 5.2 hoàn thành khi:

- dataset/checkpoint/community-feature hash và forward equivalence đều đạt;
- threshold chỉ được tính từ validation và được áp nguyên trạng lên test;
- có prediction cùng TP/FP/FN/TN cho toàn validation/test;
- danh sách target/control được chọn trước khi đọc explanation result;
- direct feature và community-risk effect có đủ cho mọi target hợp lệ;
- ablation FULL, T-CONST, N-BASE và MP-OFF chạy trên cả hai split;
- AP, ROC-AUC và sample size được báo cho từng cấu hình;
- notebook execute không có error output và mọi con số trong report khớp artifact;
- manifest kiểm tra file tồn tại, hash, schema và quan hệ ID giữa các artifact.

## 13. Ngoài phạm vi Sprint 5.2

- train lại TGAT hoặc tối ưu hyperparameter theo kết quả test;
- mở rộng hai hop hoặc đổi neighbor-sampling budget;
- thêm edge type vào model hiện tại;
- so nhiều explainer mới như PGExplainer hay Integrated Gradients;
- hiệu chỉnh xác suất phục vụ production hoặc chọn threshold theo chi phí nghiệp vụ
  chưa được cung cấp;
- xác nhận fraud ring hay xây workflow cho điều tra viên.

Các mục này chỉ nên sang sprint sau khi Sprint 5.2 xác định rõ model hiện tại đang
dựa vào node feature, thời gian hay neighbor message đến mức nào.

## 14. Cách đọc kết quả cuối

- Nếu `T-CONST` gần FULL nhưng `MP-OFF` giảm mạnh, bằng chứng về vai trò của graph
  mạnh hơn bằng chứng về khác biệt thời gian tương đối.
- Nếu cả time ablation và MP-OFF đều làm metric/logit thay đổi ổn định, có bằng chứng
  model nhạy với cả thời gian và thông tin hàng xóm.
- Nếu FP/FN chủ yếu bị đẩy bởi feature target hoặc community-risk trong khi graph
  ablation ít ảnh hưởng, ưu tiên sprint sau cho calibration/feature analysis.
- Mọi kết luận phải đi kèm split, cohort, `N`, effect size và uncertainty; không suy
  rộng tỷ lệ từ target giải thích sâu ra toàn graph.
