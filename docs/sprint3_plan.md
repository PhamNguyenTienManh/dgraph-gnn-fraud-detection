# Kế hoạch Sprint 3 — GAT và TGAT

## 1. Mục tiêu

Sprint 3 bổ sung đúng hai model: GAT và TGAT cho node-level fraud classification.
Benchmark chính dùng graph undirected để cùng direction với TGAT đã chọn.

## 2. Cấu hình chính thức

- GAT: graph undirected `structural_coalesced`, hai layer, bốn head với tổng hidden
  width 64, ReLU, hidden dropout 0,5, attention dropout 0 và fan-out `[15,10]`.
- TGAT: graph undirected `temporal_event_mirror`, một `TransformerConv`, fan-out
  `[15]`, cosine time encoding của `node_time(source) - edge_time`.
- Tên model trong code, catalog, checkpoint và báo cáo là `tgat`.
- Không giữ TGAT strict trong phạm vi Sprint 3.
- Cấu hình `[15,10]` TGAT cũ không thuộc nội dung chính và raw artifact đã được xóa.

## 3. Điều kiện so sánh

GCN, GraphSAGE, GAT và TGAT được so trên graph undirected, cùng dataset fingerprint,
node split, feature `zero_indicator` 34D, seed 42/43/44, batch 1.024, optimizer,
training budget, loss và validation-AP checkpoint selection. Test không dùng tuning.

GCN/GraphSAGE/GAT dùng structural graph hai hop; TGAT dùng temporal-event graph một
hop tương ứng một message-passing layer. Do đó bảng là so sánh pipeline end-to-end
trong cùng direction, không phải one-factor-at-a-time architecture ablation.

## 4. Kiểm tra bắt buộc

- Mỗi temporal event mirror giữ timestamp và edge type của event gốc.
- `node_time` được tính từ out-edge gốc trước khi mirror.
- TGAT loader tạo đúng một hop `[15]` và một `edge_delta` cho mỗi sampled edge.
- Loss và `pos_weight` chỉ dùng train; checkpoint chỉ dùng validation AP.
- Full result có đủ ba seed và đánh giá toàn bộ validation/test split.
- Catalog và report không trộn TGAT strict, TGAT `[15,10]` hoặc kết quả directed.

## 5. Trạng thái thực hiện

- GAT undirected full run: hoàn tất seed 42/43/44.
- TGAT undirected `[15]` full run: hoàn tất seed 42/43/44.
- TGAT strict: đã dọn khỏi code, catalog, report và raw run.
- TGAT `[15,10]`: đã bỏ khỏi nội dung chính và xóa raw run.
- Report chính: so sánh GCN, GraphSAGE, GAT và TGAT undirected.
