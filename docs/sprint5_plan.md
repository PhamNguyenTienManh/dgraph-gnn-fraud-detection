# Kế hoạch Sprint 5 — Giải thích TGAT + community-risk

## 1. Bối cảnh

Sprint trước đã so sánh các cấu hình bằng validation AP và chọn **TGAT +
community-risk** làm mô hình chính. Sprint 5 không huấn luyện detector mới. Sprint này
dùng GNNExplainer và các phép kiểm tra trực tiếp để trả lời:

> Mô hình dựa vào thông tin nào để đánh giá một node là đáng ngờ, và lời giải thích
> đó có đủ ổn định, đáng tin để sử dụng hay không?

Kết quả chỉ là lời giải thích cục bộ cho dự đoán của model trên DGraphFin. Không xem
event được chọn là bằng chứng nhân quả và không gọi graph con là fraud ring đã xác nhận.

## 2. Ba mục tiêu theo đúng thứ tự trình bày

### 2.1. GNNExplainer nói TGAT chú ý gì?

- Mô tả số event model nhìn thấy quanh mỗi target.
- Tổng hợp năm feature đứng đầu theo validation, test và bốn cohort.
- Nêu riêng community-risk đứng hạng bao nhiêu và thay đổi dự đoán ra sao khi đưa nó
  về mức nền.
- Mô tả event được xếp cao theo loại event, độ gần về thời gian (edge_delta) và
  community.

### 2.2. Kiểm chứng lời giải thích

- So thứ tự của GNNExplainer với phép đưa từng feature về mức nền.
- So event đứng đầu với phép bỏ lần lượt từng event rồi chạy model lại.
- Kiểm tra nhóm event đứng đầu bằng hai phép thử:
  1. chỉ giữ nhóm đó và xem dự đoán có gần dự đoán đầy đủ không;
  2. bỏ nhóm đó và xem độ tin của model vào lớp đang dự đoán có giảm không.
- So cùng điều kiện giữ/bỏ với ba quy tắc đơn giản: ngẫu nhiên, hàng xóm có nhiều liên
  kết và event gần nhất.
- Kiểm tra checkpoint stability và negative control với model có trọng số ngẫu nhiên.

### 2.3. Quyết định có sử dụng được không

- Chỉ đề xuất candidate risky subgraph nếu lời giải thích event đạt fidelity, ổn định
  và tốt hơn hợp lý so với các quy tắc đơn giản trên test.
- Nếu chưa đạt, vẫn báo trung thực feature/event model chú ý, nhưng không chuyển graph
  con cho người điều tra như một kết quả đáng tin độc lập.

## 3. Model, dữ liệu và phạm vi bị khóa

- Dataset: data/dgraphfin.npz, SHA-256
  95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85.
- Model: TGAT + community-risk đã chọn ở Sprint 4.
- Checkpoint chính: seed 42, SHA-256
  558db58e8c788edb8f5862b64093ef5a1a463debadd80e7e119d230ca602b61e.
- Robustness checkpoint: seed 43 và 44.
- Graph: temporal_event_mirror.
- Neighborhood: một hop, lấy tối đa 15 event; batch được materialize một lần và giữ
  nguyên trong quá trình tối ưu mask.
- Model ở chế độ eval và bị đóng băng; Sprint 5 không cập nhật trọng số model.
- Kết luận giới hạn ở node-level, full-history transductive classification và local
  post-hoc explanation.

Các hướng hai hop, GraphSAGE, time perturbation, Integrated Gradients và so TGAT
không có community-risk là công việc mở rộng, không phải điều kiện hoàn thành Sprint 5.

## 4. Chọn target

Giải thích 80 node: 40 validation và 40 test. Mỗi split có bốn cohort, mỗi cohort 10
node:

| Cohort | Quy tắc |
|---|---|
| High-score fraud | Label fraud, fraud score cao nhất |
| High-score normal | Label normal, fraud score cao nhất |
| Low-score fraud | Label fraud, fraud score thấp nhất |
| Low-score normal control | Score thấp, ghép gần theo log-degree và community size |

Danh sách target và cohort được khóa trước khi phân tích hình. Không loại node chỉ vì
neighborhood ít event hoặc hình khó đọc.

## 5. GNNExplainer trong thí nghiệm

### 5.1. Đầu vào được chấm điểm

- **Event mask:** mỗi temporal event trong local computation graph có một điểm.
- **Feature mask:** 35 đầu vào gồm 17 feature ẩn danh F01–F17, 17 cờ báo thiếu dữ liệu
  và community-risk.

F01–F17 không có tên ngữ nghĩa công khai trong DGraphFin. Báo cáo chỉ dùng đúng mã
feature, không tự gán ý nghĩa nghiệp vụ.

### 5.2. Cấu hình cuối

- Thư viện: torch_geometric.explain.GNNExplainer.
- Hai lượt giải thích độc lập: event-only và feature-only để hai mask không che lấp nhau.
- Mọi mask bắt đầu ở logit 0, tương ứng trọng số giữ lại 0,5.
- 50 epoch, learning rate 0,01.
- Feature lấy top-5 đã khóa trên validation.
- Event lấy prefix ngắn nhất theo thứ tự importance thỏa:
  sufficiency error không quá 0,05 và comprehensiveness dương.

Khởi tạo 0,5 làm mọi event/feature xuất phát ngang nhau. Với cùng frozen batch, model
và cấu hình, quá trình tối ưu là xác định; vì vậy không dùng các lần lặp chỉ khác
explainer_seed như bằng chứng stability giả tạo. Seed vẫn được lưu cho provenance.
Biến thiên có ý nghĩa được kiểm tra qua các checkpoint model và neighborhood đã khóa.

## 6. Phân tích feature

Mọi tổng hợp feature phải đọc từ sprint5_explanations_final.npz:

1. đếm tần suất feature xuất hiện trong top-5 của GNNExplainer;
2. làm riêng cho validation, test và từng cohort;
3. đưa lần lượt từng feature về mức nền, chạy model lại và xếp hạng theo mức thay đổi;
4. so top-5 và toàn bộ ranking của hai cách;
5. không dùng bất kỳ feature mask nào ngoài artifact cuối.

### Community-risk

Ngoài rank của mask, thay riêng community-risk của target bằng global train prior,
giữ nguyên graph và 34 feature còn lại, rồi lưu:

- raw logit và fraud score trước/sau;
- chênh lệch logit và score;
- tổng hợp theo validation/test và cohort.

Kết quả counterfactual được phép tích hợp khi nó có cùng dataset hash, checkpoint hash
và đúng 80 target. Artifact cuối phải tự chứa từng forward pass, mô tả baseline và kết
quả kiểm tra provenance.

## 7. Phân tích event và community

Với từng event lưu event ID, hai đầu node, timestamp, edge_delta, event type, điểm
GNNExplainer, tác động khi bỏ riêng event và thứ hạng của hai cách.

Tổng hợp:

- bao nhiêu target có từ hai event để thật sự có thể xếp hạng;
- GNNExplainer chọn event gần nhất bao nhiêu lần;
- event type nào xuất hiện nói chung và trong nhóm đứng đầu;
- event được chọn có nằm cùng Leiden community với target không.

Nếu tất cả sampled event đều nội bộ community thì phải nói rõ dữ liệu quan sát không
cho phép so sở thích edge nội bộ với edge đi ra ngoài; không diễn giải tỷ lệ 1,0 như
một thành công của explainer.

## 8. Các phép kiểm chứng

### 8.1. Kiểm tra trực tiếp từng đầu vào

- Event: bỏ lần lượt từng event và tìm event làm target-class score đổi nhiều nhất.
- Feature: đưa lần lượt từng feature về mức nền và đo thay đổi.
- Báo top-1 match cho event, Spearman và top-5 Jaccard cho feature.

Đây là phép kiểm tra độc lập tương đối, không phải ground truth tuyệt đối: bỏ từng đầu
vào có thể không biểu diễn hết tương tác giữa nhiều event/feature.

### 8.2. Kiểm tra nhóm event bằng giữ/bỏ

Một nhóm event đạt khi:

- chỉ giữ nhóm đó: target-class score lệch không quá 0,05 so với graph đầy đủ;
- bỏ nhóm đó: target-class score giảm.

Áp dụng cùng quy tắc chọn prefix thích ứng cho GNNExplainer và ba baseline. Báo cả số
event phải giữ; một phương pháp phải giữ gần toàn bộ neighborhood không được xem là
tạo giải thích gọn.

### 8.3. Negative control

Giữ nguyên kiến trúc nhưng thay trọng số đã học bằng ba bộ trọng số ngẫu nhiên trên
năm validation node có nhiều event nhất. So ranking của chúng với model đã train bằng
Spearman, top-3 Jaccard và exact ranking match. Model ngẫu nhiên không “rank random”
theo kiểu bốc thăm; nó vẫn tính toán nhất quán, nhưng chưa học tín hiệu từ dữ liệu.

### 8.4. Stability qua checkpoint

Trên 12 validation node đã khóa, chạy cùng protocol với checkpoint seed 42/43/44 và
đo edge Jaccard, feature top-5 Jaccard và feature Spearman. Neighborhood và ID mapping
phải được giữ/đối chiếu rõ ràng.

## 9. Bộ hình cuối bắt buộc

| File | Nội dung |
|---|---|
| 01_neighborhood_size.png | Số event quanh 80 target |
| 02_feature_patterns.png | Feature top-5 của GNNExplainer so với đưa từng feature về nền |
| 03_community_risk_counterfactual.png | Tác động trực tiếp của community-risk theo split/cohort |
| 04_event_explanation_checks.png | Kiểm tra event đứng đầu và kiểm tra giữ/bỏ so baseline |

Manifest và gate phải kiểm tra đúng bốn file trên tồn tại, không chấp nhận đường dẫn
cũ hoặc chỉ kiểm tra một danh sách được ghi sẵn trong JSON.

## 10. Luồng report và Notebook 06

Report và notebook dùng cùng artifact cuối và đi theo thứ tự:

1. model nhìn thấy gì;
2. GNNExplainer chọn feature nào;
3. community-risk đóng góp thế nào;
4. GNNExplainer chọn event nào;
5. lời giải thích được kiểm tra ra sao;
6. kết luận về GNNExplainer và các hướng cải tiến.

Chỉ trình bày protocol và kết quả cuối. Không kể lịch sử chạy hoặc quyết định sửa lỗi
trong tài liệu dành cho người đọc. Thuật ngữ phải được giải thích ngay khi xuất hiện.

## 11. Artifact bàn giao

- artifacts/metrics/sprint5_explainer_final.json: 80 lời giải thích đã làm giàu,
  summary kiểm chứng, model insight và community-risk counterfactual.
- artifacts/metrics/sprint5_explanations_final.npz: mask cuối và tác động bỏ từng
  event/feature.
- artifacts/metrics/sprint5_explanation_final.csv: bảng phẳng 80 target.
- artifacts/metrics/sprint5_model_insights_final.json: tổng hợp feature, event,
  community-risk và community relation.
- artifacts/metrics/sprint5_explainer_manifest.json: provenance, danh sách artifact
  và gate.
- bốn hình trong artifacts/figures/sprint5/.
- notebooks/06_gnn_explainer.ipynb.
- docs/sprint5_report.md.

## 12. Gate hoàn thành

Sprint 5 hoàn thành khi:

- dataset/checkpoint hash và adapter logit equivalence đạt;
- có đúng 40 validation + 40 test target;
- artifact cuối dùng mask khởi tạo trung tính 0,5;
- đủ raw mask, direct event/feature effect và community-risk counterfactual cho 80 target;
- có tổng hợp feature/event/community theo split và cohort;
- negative control và cross-checkpoint stability có kết quả;
- bốn hình cuối tồn tại; không còn tham chiếu bộ hình cũ;
- notebook đã execute, không có error output và đọc đúng artifact cuối;
- mọi số liệu trong report khớp artifact;
- danh sách candidate risky subgraph để trống nếu bằng chứng event trên test chưa vượt
  baseline một cách thuyết phục.

Scientific outcome không cần “đẹp” để gate kỹ thuật đạt. Một kết quả cho thấy
GNNExplainer chưa đủ mạnh để chuyển graph con cho điều tra viên vẫn là kết quả Sprint 5
hợp lệ nếu được báo đầy đủ và trung thực.

## 13. Công việc mở rộng

1. Khóa threshold trên validation, tách false positive/false negative và kiểm tra trực
   tiếp đầu vào nào đã đẩy model tới dự đoán sai.
2. Thay đổi riêng timestamp, hàng xóm và message passing để kiểm tra TGAT học thời
   gian và cấu trúc graph đến đâu.
3. Báo riêng target có từ hai event và so các phương pháp ở cùng top-1, top-2, top-3
   thay vì cho phép giữ gần toàn bộ neighborhood.

## 14. Tài liệu phương pháp

- Ying et al. (2019), *GNNExplainer: Generating Explanations for Graph Neural
  Networks*, NeurIPS.
- PyTorch Geometric, torch_geometric.explain.GNNExplainer.
- Huang et al. (2022), *DGraph: A Large-Scale Financial Dataset for Graph Anomaly
  Detection*, NeurIPS Datasets and Benchmarks.
