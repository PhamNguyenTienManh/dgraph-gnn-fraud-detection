GRAPH NEURAL NETWORKS FOR DYNAMIC FRAUD-RING DETECTION ON THE DGRAPH
DATASET 

Sprint 1 --- Chuẩn bị bộ dữ liệu và Graph Pipeline Tìm hiểu bộ
dữ liệu DGraph, bao gồm cấu trúc đồ thị, ý nghĩa của node, edge,
timestamp, node features và label. Xây dựng pipeline đọc dữ liệu DGraph
và chuyển đổi sang định dạng đồ thị phục vụ cho các mô hình Graph Neural
Network. Khảo sát các tập train, validation và test đã được cung cấp sẵn
trong bộ dữ liệu. Thực hiện sampling hoặc xây dựng subgraph phù hợp với
tài nguyên tính toán của máy cá nhân để phục vụ quá trình huấn luyện.
Kiểm tra tính chính xác của pipeline thông qua thống kê số lượng node,
edge, số chiều đặc trưng và phân bố nhãn trước khi tiến hành huấn luyện
mô hình. 

Sprint 2 --- Xây dựng các mô hình GNN Baseline (GCN và
GraphSAGE) Cài đặt và huấn luyện mô hình GCN trên bộ dữ liệu DGraph.
Thiết lập pipeline huấn luyện, validation và inference; đánh giá hiệu
năng bằng các chỉ số AUC và Average Precision (AP). Cài đặt và huấn
luyện mô hình GraphSAGE với cùng bộ dữ liệu và điều kiện thực nghiệm
nhằm đảm bảo tính công bằng khi so sánh. Chuẩn hóa các tham số huấn
luyện (learning rate, batch size, optimizer, epoch...) giữa các mô hình
baseline. So sánh kết quả giữa GCN và GraphSAGE; phân tích khả năng khai
thác thông tin cấu trúc đồ thị của từng mô hình. Lưu trữ mô hình và kết
quả thực nghiệm phục vụ cho quá trình benchmark ở các sprint tiếp theo.

Sprint 3 --- Triển khai GAT, TGAT và Benchmark Cài đặt và huấn luyện mô
hình GAT nhằm đánh giá hiệu quả của cơ chế Attention trong Graph Neural
Network. Cài đặt và huấn luyện mô hình TGAT để khai thác thông tin thời
gian (Temporal Information) trong bộ dữ liệu DGraph. Thực hiện benchmark
giữa GCN, GraphSAGE, GAT và TGAT trên cùng bộ dữ liệu và cùng bộ chỉ số
đánh giá. So sánh kết quả thực nghiệm; phân tích ưu, nhược điểm của từng
kiến trúc GNN đối với bài toán Dynamic Fraud-Ring Detection. Lựa chọn mô
hình có hiệu năng tốt nhất làm baseline cho bước cải tiến.

Sprint 4 --- Cải tiến mô hình phát hiện Fraud-Ring Nghiên cứu và triển
khai một hướng cải tiến nhằm nâng cao hiệu quả của mô hình baseline,
chẳng hạn Community Detection hoặc Risk Propagation. Tích hợp thông tin
Community Detection hoặc Risk Propagation vào pipeline của mô hình và
đánh giá khả năng cải thiện hiệu năng. So sánh kết quả trước và sau khi
áp dụng phương pháp cải tiến thông qua các chỉ số AUC và Average
Precision (AP). Thử nghiệm phương pháp Explainable Graph AI (ví dụ
GNNExplainer) nhằm xác định các Risky Subgraph để hỗ trợ giải thích kết
quả dự đoán của mô hình. Phân tích mức độ đóng góp của từng phương pháp
cải tiến đối với bài toán phát hiện gian lận trên đồ thị động.

Sprint 5 --- Tổng hợp kết quả và Hoàn thiện đồ án Tổng hợp toàn bộ kết
quả thực nghiệm của các mô hình GCN, GraphSAGE, GAT, TGAT và các phương
pháp cải tiến. So sánh hiệu năng giữa các mô hình; phân tích ưu điểm,
hạn chế và khả năng ứng dụng của từng phương pháp đối với bài toán
Dynamic Fraud-Ring Detection. Hoàn thiện báo cáo, biểu đồ, bảng số liệu
và slide trình bày kết quả nghiên cứu. Đánh giá mức độ hoàn thành mục
tiêu đề tài; đề xuất các hướng phát triển trong tương lai như mở rộng
sang các Temporal GNN tiên tiến hơn, kết hợp Community Detection, Risk
Propagation hoặc Explainable Graph AI để nâng cao hiệu quả và khả năng
giải thích của mô hình.
