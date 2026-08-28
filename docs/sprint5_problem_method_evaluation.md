# Sprint 5 — Problem, method và evaluation

Sprint 4 cho thấy **TGAT + community-risk** là cấu hình tốt nhất trong các model đã thử. Sprint 5 giữ nguyên model đó và trả lời hai câu hỏi tiếp theo:

1. Model dựa vào feature và event nào khi đánh giá một node?
2. Lời giải thích do GNNExplainer tạo ra có đủ đáng tin để sử dụng không?

Trong tài liệu này, **target node** là node đang được giải thích. **Event** là một quan hệ, tức một cạnh nối target với một node lân cận trong graph cục bộ. Số liệu và hình minh họa đầy đủ hơn nằm trong `sprint5_report.md`.

## 1. GNNExplainer được dùng như thế nào trong bài?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **1.1. GNNExplainer tạo lời giải thích bằng cách nào?** | Giữ nguyên trọng số TGAT đã học. Với mỗi target, đặt một “nút điều chỉnh” hay mask lên từng feature hoặc event. GNNExplainer thay đổi các mask để tìm phần đầu vào có thể giữ lại dự đoán gần với dự đoán ban đầu. Sau đó feature và event được xếp từ điểm mask cao xuống thấp. | Tạo được lời giải thích feature và event cho đủ `80/80` target. | Thứ hạng của GNNExplainer là một giả thuyết về phần đầu vào model đang dựa vào, không phải đáp án đúng có sẵn. Vì vậy Sprint 5 còn phải kiểm tra lại thứ hạng này bằng các phép can thiệp trực tiếp vào đầu vào. |
| **1.2. Vì sao feature và event được giải thích riêng?** | Chạy hai lượt độc lập: một lượt chỉ tối ưu mask feature, một lượt chỉ tối ưu mask event. Tất cả mask bắt đầu bằng `0,5`, nghĩa là mọi đầu vào xuất phát ngang nhau. | Mỗi target có một thứ hạng cho 35 feature và một thứ hạng riêng cho các event xung quanh. | Tách hai lượt giúp mask feature không bù cho mask event, hoặc ngược lại. Khởi tạo bằng `0,5` chỉ tạo điểm xuất phát công bằng; nó không ép kết quả cuối phải giống nhau và không thay đổi trọng số của TGAT. |

## 2. Cách chọn target node

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **2.1. Target được lấy từ đâu?** | Chỉ lấy node từ validation và test, vì các node này không được dùng để cập nhật trọng số model. Trong mỗi tập, chia target thành bốn nhóm: fraud điểm cao, normal điểm cao, fraud điểm thấp và normal điểm thấp. | Validation có `10` node trong mỗi nhóm, tổng cộng `40` node. Test cũng có `10` node trong mỗi nhóm, tổng cộng `40` node. Vì vậy toàn bộ thí nghiệm gồm `80` target và mỗi nhóm có `20` node. | Cách chọn này bao phủ cả trường hợp model phát hiện đúng, cảnh báo nhầm và bỏ sót. Nó phù hợp để xem lời giải thích thay đổi theo kiểu dự đoán. |
| **2.2. Các tỷ lệ trên 80 target có đại diện cho toàn bộ graph không?** | Không lấy mẫu ngẫu nhiên theo tỷ lệ tự nhiên của dữ liệu; bốn nhóm được cố ý lấy số lượng bằng nhau. | Những con số như `17/80` hoặc `73/80` chỉ mô tả tập target đã chọn. | Không được suy rộng rằng cùng tỷ lệ đó sẽ xuất hiện trên hàng triệu node còn lại. Mẫu 80 node dùng để so sánh các tình huống dự đoán, không dùng để ước lượng tỷ lệ của toàn bộ DGraphFin. |

## 3. Model thực sự nhìn thấy những đầu vào nào?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **3.1. Phần feature gồm những gì?** | Giải thích 35 đầu vào của target: 17 feature ẩn danh `F01–F17`, 17 cờ cho biết feature tương ứng bị thiếu và một feature community-risk từ Sprint 4. | Mỗi lời giải thích feature xếp hạng đủ `35` đầu vào. | DGraphFin không công bố ý nghĩa nghiệp vụ của `F01–F17`, nên Sprint 5 chỉ có thể nói model thường chú ý đến mã feature nào; không tự gán cho chúng ý nghĩa như thu nhập hay hành vi giao dịch. |
| **3.2. Phần event quanh target lớn đến đâu?** | Dùng neighborhood một hop và cho phép lấy tối đa 15 event trực tiếp quanh target. Sau khi cố định mẫu, thống kê số event thực tế model nhận được. | Trong 80 target, `41` node chỉ có một event và `68` node có không quá ba event; node nhiều nhất có tám event. Chỉ `39` target có từ hai event trở lên để thật sự so sánh event nào đứng đầu. | Neighborhood nhỏ làm bài toán xếp hạng event khá hạn chế. Với node chỉ có một event, mọi phương pháp đều buộc phải chọn event đó; vì vậy kết quả tốt trên những node này không cho thấy GNNExplainer hơn cách chọn đơn giản. |

## 4. GNNExplainer cho biết model chú ý đến feature nào?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **4.1. Feature nào thường xuất hiện trong top-5?** | Với từng target, lấy năm feature có mask cao nhất rồi đếm tần suất xuất hiện trên 80 lời giải thích. | Xuất hiện nhiều nhất là `F03` (`46/80`), `F07` (`42/80`), `F11` (`41/80`), `F10` (`38/80`) và `F17` (`34/80`). | Đây là nhóm feature model thường dựa vào trong mẫu đã chọn. Kết quả phù hợp để mô tả xu hướng chung, nhưng chưa đủ để khẳng định thứ hạng chính xác của hai feature gần nhau tại một node riêng lẻ. |
| **4.2. Community-risk quan trọng đến đâu?** | Ghi nhận hạng của community-risk, sau đó thay riêng giá trị của nó bằng fraud rate chung của tập train trong khi giữ nguyên graph và 34 feature còn lại. | Community-risk vào top-5 ở `17/80` target, đứng hạng 1 ở `2/80` và có hạng trung vị `11/35`. Trong nhóm fraud điểm cao, nó vào top-5 ở `11/20`; khi đưa về mức nền, điểm số nội bộ của model giảm trung bình `0,257`. | Community-risk không quyết định mọi dự đoán. Nó có vai trò rõ nhất đối với một phần các node fraud mà model chấm điểm cao, đúng với việc Sprint 4 chỉ ghi nhận một mức cải thiện nhỏ chứ không phải model hoàn toàn phụ thuộc vào feature này. |

## 5. Kiểm tra chất lượng GNNExplainer

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **5.1. Thứ hạng feature có khớp với tác động thật lên dự đoán không?** | Lần lượt đưa riêng từng feature về mức nền, chạy lại model và xếp hạng theo mức dự đoán thay đổi. So thứ hạng trực tiếp này với GNNExplainer bằng Spearman trên đủ 35 feature và Jaccard trên hai nhóm top-5. | Trung vị Spearman là `0,467`. Trung vị Jaccard top-5 là `0,667`, thường tương ứng với bốn feature chung trong hai nhóm top-5. Phép can thiệp trực tiếp cũng thường chọn `F07`, `F03`, `F10`, `F11` và `F17`. | Hai cách khá đồng ý về nhóm feature nổi bật, nhưng chỉ đồng ý ở mức vừa phải về thứ tự đầy đủ. Vì thế có thể tin hơn vào nhóm feature lặp lại nhiều lần, còn chênh lệch một vài bậc tại một node không nên được xem là kết luận chắc chắn. |
| **5.2. Event đứng đầu có khớp với event gây tác động mạnh nhất không?** | Trên 39 target có ít nhất hai event, lần lượt bỏ từng event rồi xem event nào làm độ tin của model vào lớp đang dự đoán giảm nhiều nhất. So event đó với top-1 của GNNExplainer và với quy tắc đơn giản chọn event gần thời điểm target nhất. | Khi gộp validation và test, GNNExplainer khớp `19/39`, còn event gần nhất khớp `16/39`. Riêng test, GNNExplainer khớp `8/22`, thấp hơn event gần nhất với `10/22`. | GNNExplainer có lợi thế nhỏ khi gộp hai tập nhưng không giữ được lợi thế trên test. Phép bỏ từng event cũng không phải đáp án tuyệt đối vì nhiều event có thể tương tác với nhau, nhưng kết quả này cho thấy chưa nên tin chắc event đứng đầu của GNNExplainer. |
| **5.3. Hai phép kiểm tra giữ và bỏ trả lời điều gì?** | **Phép giữ:** chỉ giữ nhóm được giải thích chọn và kiểm tra dự đoán có lệch không quá `0,05` so với khi dùng đầy đủ đầu vào. **Phép bỏ:** loại chính nhóm đó và kiểm tra độ tin của model vào lớp đang dự đoán có giảm hay không. Với feature, nhóm được kiểm tra là top-5. Với event, thử top-1, top-2, ... và lấy nhóm ngắn nhất đạt cả hai điều kiện. | Một lời giải thích chỉ được tính là đạt nếu đồng thời qua cả phép giữ và phép bỏ. | Phép giữ kiểm tra nhóm được chọn có đủ thông tin để tái hiện dự đoán hay không. Phép bỏ kiểm tra model có thật sự mất một phần căn cứ khi nhóm đó biến mất hay không. Chỉ làm một trong hai phép chưa đủ để kết luận nhóm được chọn có ý nghĩa. |
| **5.4. GNNExplainer đạt kết quả ra sao so với các cách chọn đơn giản?** | Áp dụng cùng quy trình giữ/bỏ cho thứ hạng GNNExplainer và ba thứ hạng event đơn giản: xáo ngẫu nhiên bằng seed cố định, ưu tiên hàng xóm có nhiều liên kết và ưu tiên event gần nhất. | Top-5 feature đạt cả hai phép ở `54/80` target. Nhóm event của GNNExplainer đạt `73/80`; ba cách đơn giản lần lượt đạt `70/80`, `71/80` và `70/80`. Trong ít nhất một nửa số trường hợp đạt, các phương pháp phải giữ toàn bộ event quanh target. | `73/80` cho thấy thường có thể tìm một nhóm event tái hiện dự đoán, nhưng không chứng minh GNNExplainer vượt trội: baseline cũng đạt 70–71 node và nhóm được giữ thường không nhỏ hơn neighborhood ban đầu. |

## 6. Kiểm tra độ ổn định và vai trò của trọng số đã học

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **6.1. Nếu train lại TGAT từ điểm khởi đầu khác, lời giải thích feature có thay đổi hoàn toàn không?** | Chạy cùng quy trình trên 12 node validation với ba checkpoint TGAT được train bằng seed `42`, `43` và `44`. So toàn bộ thứ hạng feature và nhóm top-5 giữa các checkpoint. | Spearman feature trung vị đạt `0,801`; Jaccard top-5 trung vị đạt `0,667`. | Các lần train không tạo ra thứ hạng giống hệt nhau, nhưng phần lớn vẫn giữ cùng xu hướng feature. Đây là bằng chứng rằng kết luận ở cấp nhóm feature tương đối ổn định qua ba lần train. |
| **6.2. Nếu model chưa được train, nó có tạo thứ hạng event giống model đã học không?** | Giữ nguyên kiến trúc TGAT nhưng thay trọng số đã học bằng ba bộ trọng số ngẫu nhiên. Chạy trên năm node validation có nhiều event nhất và so thứ hạng với model đã train. | Qua 15 lần so sánh, Spearman event trung vị chỉ khoảng `0,004`, Jaccard top-3 bằng `0` và không có thứ tự đầy đủ nào trùng nhau. | GNNExplainer không tạo cùng một thứ hạng bất kể model đã học gì; lời giải thích có phản ánh trọng số TGAT đã học. Tuy nhiên, phép kiểm tra này chỉ loại trừ một lỗi cơ bản, không chứng minh thứ hạng event của model đã train là chính xác. |

## 7. Kết quả có thể được sử dụng đến mức nào?

| Câu hỏi | Cách làm | Kết quả | Ý nghĩa |
|---|---|---|---|
| **7.1. Có nên chuyển graph con được chọn cho người điều tra không?** | Chỉ chấp nhận graph con nếu nó qua phép giữ/bỏ, đủ gọn, ổn định và tốt hơn hợp lý so với các cách chọn đơn giản trên test. | Không có graph con nào được đưa vào danh sách đề xuất. | Quyết định không phát hành graph con là phù hợp vì event ranking chưa hơn baseline trên test và nhiều target có neighborhood quá nhỏ để đánh giá chắc chắn. |
| **7.2. Sprint 5 kết luận được gì và chưa kết luận được gì?** | Đối chiếu kết quả với đúng đầu vào và các phép can thiệp đã thực hiện. | Sprint 5 xác định được xu hướng feature, vai trò không đồng đều của community-risk và giới hạn của lời giải thích event. Thí nghiệm chưa thay đổi riêng thời gian event, chưa mở rộng sang hai hop và model hiện tại không dùng `edge_type`. Cả `179/179` event quan sát được cũng nằm cùng Leiden community với target. | Có thể dùng kết quả feature như gợi ý ở cấp nhóm khi đi kèm phép kiểm tra trực tiếp. Chưa thể kết luận TGAT không học thời gian hoặc không học cấu trúc graph; cũng không thể nói model ưu tiên loại quan hệ nào hay gọi graph con là fraud ring. |

## Kết luận ngắn

- GNNExplainer cho thấy `F03`, `F07`, `F11`, `F10` và `F17` thường được TGAT + community-risk ưu tiên; phép đưa từng feature về nền xác nhận xu hướng tương tự.
- Community-risk có đóng góp rõ nhất ở một phần nhóm fraud điểm cao, nhưng không chi phối mọi dự đoán.
- Hai phép giữ và bỏ cho thấy GNNExplainer thường tìm được một nhóm đầu vào tái hiện dự đoán, nhưng phần event chưa gọn hơn hoặc tốt hơn rõ ràng so với các cách chọn đơn giản.
- Lời giải thích feature tương đối ổn định qua ba checkpoint, và phép dùng trọng số ngẫu nhiên cho thấy kết quả không xuất hiện độc lập với những gì TGAT đã học.
- Vì bằng chứng về thứ hạng event còn yếu, Sprint 5 không đề xuất graph con cho người điều tra. Kết quả đáng sử dụng nhất hiện tại là xu hướng feature ở cấp nhóm node, kèm phép kiểm tra trực tiếp.
