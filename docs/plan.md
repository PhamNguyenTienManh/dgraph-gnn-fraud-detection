GRAPH NEURAL NETWORKS FOR FRAUD DETECTION ON THE DGRAPH
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
kiến trúc GNN đối với bài toán phát hiện gian lận cấp node trên graph. Lựa chọn mô
hình có hiệu năng tốt nhất làm baseline cho bước cải tiến.

Sprint 4 --- Cải tiến mô hình phát hiện gian lận Nghiên cứu và triển
khai một hướng cải tiến nhằm nâng cao hiệu quả của mô hình baseline,
chẳng hạn Community Detection hoặc Risk Propagation. Tích hợp thông tin
Community Detection hoặc Risk Propagation vào pipeline của mô hình và
đánh giá khả năng cải thiện hiệu năng. So sánh kết quả trước và sau khi
áp dụng phương pháp cải tiến thông qua các chỉ số AUC và Average
Precision (AP). Phân tích mức độ đóng góp của từng phương pháp cải tiến
đối với bài toán phát hiện gian lận trên graph.

Sprint 5 --- Explainable Graph AI và Hoàn thiện đồ án Thử nghiệm
GNNExplainer nhằm xác định các Risky Subgraph và hỗ trợ giải thích kết
quả dự đoán của mô hình. Tổng hợp kết quả các sprint, hoàn thiện báo
cáo, biểu đồ và slide trình bày.
