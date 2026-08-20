# Sprint 4 — Problem, method và evaluation

Tài liệu này tóm tắt các câu hỏi chính của Sprint 4, cách thực hiện và ý nghĩa của
kết quả. Số liệu đầy đủ hơn được trình bày trong `sprint4_report.md`.

## 1. Dùng graph nào và chạy Leiden như thế nào?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **1.1. Dữ liệu đầu vào cho Leiden là gì?** | Kiểm tra SHA-256, số node và số cạnh của `dgraphfin.npz`. Hai cạnh ngược chiều giữa cùng một cặp node được gộp thành một cạnh vô hướng duy nhất. Leiden chỉ nhận cấu trúc graph, không nhận nhãn fraud. | Graph có `3.700.550` node và `4.300.999` cạnh có hướng. Sau khi gộp còn `3.997.260` cặp cạnh vô hướng, tương đương `7.994.520` cạnh hai chiều. | Community được tạo từ quan hệ giữa các node, không được tạo dựa trên việc node nào là fraud. Vì vậy bước chia community không bị ảnh hưởng bởi nhãn. |
| **1.2. Leiden được chạy bằng cấu hình nào?** | Dùng hàm `community_leiden` của thư viện `python-igraph 1.0.0`, tối ưu modularity với resolution `1.0`, seed `42` và `2` iterations. Kết quả `node_id → community_id` được lưu kèm cấu hình và hash của file. | Lần chạy gốc mất khoảng `1,34` giây để dựng graph và `29,08` giây để chạy Leiden; peak RAM khoảng `742,65 MiB`. | Notebook không tự cài đặt Leiden. Việc lưu seed, cấu hình và hash giúp phát hiện khi dữ liệu hoặc kết quả community bị thay đổi. |

## 2. Các community tìm được có hợp lý không?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **2.1. Leiden tạo ra bao nhiêu community và kích thước ra sao?** | Thống kê số community, kích thước nhỏ nhất, trung vị, lớn nhất, số community chỉ có một node và tỷ lệ của community lớn nhất so với toàn graph. | Có `1.739` community. Kích thước nhỏ nhất là `99`, trung vị `1.964` và lớn nhất `16.609` node. Không có community chỉ gồm một node; community lớn nhất chiếm khoảng `0,45%` toàn graph. | Kết quả không bị chi phối bởi một community quá lớn và cũng không bị chia vụn thành rất nhiều community một node. |
| **2.2. Các node trong cùng community có thực sự liên kết chặt không?** | Đo modularity, coverage, conductance và internal-edge ratio. Coverage và internal-edge ratio cao cho biết phần lớn cạnh nằm trong community; conductance thấp cho biết community có ít cạnh nối ra ngoài. | Modularity đạt `0,994726`, coverage `0,995465`, conductance trung vị `0,003653` và internal-edge ratio trung vị `0,996347`. | Graph có cấu trúc community rõ. Tuy nhiên, các chỉ số này chỉ đánh giá cách chia graph; chúng chưa chứng minh community có ích cho việc phát hiện fraud. |

## 3. Fraud có tập trung trong một số community không?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **3.1. Risky community được chọn như thế nào?** | Chỉ dùng nhãn train. Một community phải có ít nhất 20 node, 20 train label, 2 train fraud, fraud lift từ `2` trở lên và internal-edge ratio từ `0,5` trở lên. Sau đó xếp hạng và khóa top 5 trước khi xem validation. | Có `51` community đạt điều kiện. Top 5 được chọn là `54, 286, 110, 296, 1544`. | Validation và test không tham gia chọn community. Điều này giúp kiểm tra xem rule tạo từ train có còn đúng trên dữ liệu chưa dùng để lựa chọn hay không. |
| **3.2. Các community được chọn có tiếp tục nhiều fraud trên validation không?** | Sau khi khóa danh sách, dùng validation để tính fraud rate và fraud lift. Ngoài ra, so sánh thứ hạng fraud rate giữa train và validation trên các community có ít nhất 20 label ở cả hai tập. | Top 5 có `18/368` validation node là fraud, tương ứng fraud rate `4,89%` và lift `3,866×`; nhưng chỉ chứa khoảng `0,77%` tổng fraud validation. Trên `1.704` community đủ label, Spearman giữa train và validation fraud rate chỉ là `0,089272`. | Một số community có mật độ fraud cao, nhưng phạm vi bao phủ nhỏ và thứ hạng giữa train–validation chưa ổn định. Vì vậy đây chỉ là vùng nên ưu tiên kiểm tra, không phải bằng chứng rằng mọi node trong community đều gian lận. |

## 4. Community feature được tạo thế nào để tránh label leakage?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **4.1. Mỗi node nhận những community feature nào?** | Tạo năm feature không dùng nhãn: log kích thước community, log số cạnh nội bộ, mật độ cạnh nội bộ, tỷ lệ cạnh nội bộ của node và conductance. Feature community-risk là fraud rate của community tính chỉ từ train label. Các feature được chuẩn hóa bằng mean/std của train nodes. | Mỗi node có 5 structural features và 1 community-risk feature. Community có dưới 20 train label nhận fraud rate chung của tập train thay vì một tỷ lệ riêng thiếu ổn định. | Structural features mô tả hình dạng community; community-risk mô tả mức fraud đã quan sát trong train. Community ID không được đưa trực tiếp vào model vì nó chỉ là mã định danh. |
| **4.2. Vì sao train node dùng leave-one-out và risk chỉ được nối sau message passing?** | Khi tính risk cho một train node, loại nhãn của chính node đó khỏi fraud count và labeled count. Risk không được truyền qua các lớp tổng hợp hàng xóm; nó chỉ được nối với embedding của node ngay trước classifier. | Manifest của feature xác nhận train dùng leave-one-out, validation/test chỉ dùng train label và risk được tích hợp sau message passing. | Leave-one-out ngăn model nhìn trực tiếp đáp án của node train. Nối risk sau message passing còn ngăn nhãn của node đi vòng qua feature của hàng xóm rồi quay trở lại node đó. |

## 5. Community feature có giúp TGAT tốt hơn không?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **5.1. Bốn cấu hình A/B/C/D được so sánh ra sao?** | A là TGAT gốc; B thêm 5 structural features; C chỉ thêm community-risk; D thêm cả hai nhóm. Tất cả dùng cùng split, seed `42/43/44`, cấu hình train và cách chọn checkpoint bằng validation AP. | Validation AP trung bình của A/B/C/D lần lượt là `0,040787`, `0,040429`, `0,041467`, `0,041054`. C cao nhất và tốt hơn A ở cả `3/3` seed nên được chọn trước khi xem test. | Bốn cấu hình giúp tách riêng tác dụng của structural features và community-risk. Kết quả validation cho thấy community-risk hữu ích hơn năm structural features đang dùng. |
| **5.2. Kết quả cuối trên test là gì?** | Khóa checkpoint trước khi chạy test và không thay đổi model sau đó. So sánh AP và ROC-AUC của các checkpoint đã khóa. | A đạt test AP `0,043885`; B `0,043608`; C `0,044336`; D `0,044329`. C cao hơn A `0,000451`, tương đương khoảng `1,03%`, và cao hơn ở `3/3` seed. Test ROC-AUC của C tăng từ `0,784920` lên `0,785328`. | Năm structural features chưa giúp TGAT. Community-risk mang lại cải thiện nhỏ nhưng nhất quán. D gần như bằng C, nên chưa có bằng chứng rằng structural features đem lại thêm lợi ích khi đã có community-risk. |

## 6. Có thể kết luận đến mức nào?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **6.1. Pipeline đã hạn chế việc dùng test để chọn model như thế nào?** | C được chọn bằng validation AP. Trước test, notebook lưu hash của 12 checkpoint A/B/C/D. Test được chạy một lần trên các checkpoint đã khóa và không dùng để bắt đầu một vòng chỉnh model mới. | C là cấu hình được validation chọn và cũng cao hơn A trên test. Kết quả test của B/D chỉ dùng để mô tả, không dùng để đổi winner hoặc train lại. | Kết luận chính dựa trên C, không dựa trên việc nhìn test rồi chọn cấu hình có kết quả đẹp nhất. |
| **6.2. Sprint 4 chưa trả lời được những gì?** | Đối chiếu phạm vi dữ liệu và các thí nghiệm đã thực hiện. | Thí nghiệm mới dùng một dataset, một resolution và một seed cho Leiden, cùng ba seed khi train TGAT. Chưa so với Louvain, random community, nhiều resolution hoặc kiểm định thống kê. Model dùng cấu trúc toàn graph đã có và không áp dụng temporal cutoff. | Có thể kết luận community-risk hỗ trợ nhẹ cho TGAT trên thiết lập hiện tại. Chưa thể kết luận đây là dự báo fraud trong tương lai, Leiden luôn là lựa chọn tốt nhất hoặc risky community chính là một fraud ring có phối hợp. |

## Kết luận ngắn

- Leiden tìm được `1.739` community có liên kết nội bộ rõ.
- Một số community có fraud lift cao, nhưng chỉ bao phủ một phần nhỏ fraud và tín hiệu
  chưa ổn định mạnh giữa train–validation.
- Community-risk được tạo chỉ từ train label và được tích hợp sau message passing để
  tránh label leakage.
- Structural features chưa cải thiện TGAT; community-risk giúp test AP tăng khoảng
  `1,03%` và cải thiện ở cả ba seed.
- Kết quả cho thấy community-risk là một feature bổ sung hữu ích ở mức nhỏ, không thay
  thế TGAT và không đủ để khẳng định đã phát hiện một đường dây gian lận.
