# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hoàng Nam
**Nhóm:**  5changlinhngulam
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> Góc giữa 2 vector nhỏ, cho thấy 2 đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa (context) hoặc từ khóa, tùy thuộc vào loại thuật toán sinh vector (embedding model).

**Ví dụ có độ tương tự CAO:**

- Câu A: "Shopee hoàn tiền trong 7 ngày."
- Câu B: "Thời gian xử lý hoàn trả tiền của Shopee là một tuần."
- Tại sao tương đồng: Khác từ vựng nhưng cùng một ý nghĩa ngữ cảnh (7 ngày = một tuần, hoàn trả tiền = hoàn tiền).

**Ví dụ có độ tương tự THẤP:**

- Câu A: "Chính sách miễn phí vận chuyển cho thành viên Shopee."
- Câu B: "Các mặt hàng bị cấm buôn bán trên nền tảng."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (Vận chuyển vs. Hàng cấm).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> Bởi vì cosine similarity chỉ quan tâm đến *hướng* (góc) của vector (tức là ngữ nghĩa), chứ không bị ảnh hưởng bởi *độ lớn* (chiều dài) của vector (thường bị ảnh hưởng bởi độ dài của văn bản, số lượng từ lặp lại).

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> *Trình bày phép tính:* Bước tiến (step) của mỗi chunk là: 500 - 50 = 450. Số chunk = ceil(10,000 / 450) = 23 chunks.
> *Đáp án:* Khoảng 23 chunks. (Chính xác: 22 chunks đầu dài 500, chunk cuối cùng chứa phần còn lại).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> *Viết 1-2 câu:* Bước tiến giảm xuống (500 - 100 = 400), dẫn đến số chunk TĂNG lên (khoảng 25 chunks). Muốn tăng overlap để tránh việc một ý hay một câu quan trọng bị cắt đôi gãy gập giữa 2 chunk, giúp LLM có đầy đủ bối cảnh (context) ở đoạn chuyển tiếp.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

> Dùng regex `(?<=[.!?])\s+|\.\n` để ngắt chính xác tại điểm cuối của câu (dấu chấm, phẩy, chấm hỏi đi kèm khoảng trắng). Xử lý edge case: Bỏ đi khoảng trắng thừa ở cuối mỗi mảnh con, lọc các mảnh trống để đảm bảo mảng trả về luôn có ý nghĩa.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

> Thuật toán đệ quy. **Base case:** Nếu text ngắn hơn `chunk_size` hoặc không còn `separators`, lập tức trả về `text`. Nếu không, cắt text bằng `separator` đầu tiên, nối các mảnh con lại cho đến khi chạm `chunk_size`. Nếu có mảnh con bị vượt giới hạn, gọi đệ quy chính nó với bộ `separators` kế tiếp nhỏ hơn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

> Khởi tạo `in-memory` store dưới dạng một list of dictionaries. `add_documents` sẽ tạo embedding cho nội dung và append vào list. Hàm `search` sẽ dùng `compute_similarity` để quét toàn bộ mảng tính điểm Cosine, sau đó sort (sắp xếp giảm dần) và lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

> **Lọc trước (Pre-filtering)**: Chạy một vòng lặp loại bỏ (filter) các dict không khớp `metadata` TRƯỚC, rồi mới tính điểm Cosine cho những record còn lại (giúp tối ưu tốc độ tính toán). `delete_document` sử dụng thao tác tái cấu trúc mảng: chỉ giữ lại các record mà `metadata['doc_id']` KHÔNG khớp với id truyền vào.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

> Nối mảng content lấy từ `store.search` bằng ký tự `\n\n` để tạo thành Context. Cấu trúc Prompt cơ bản: `"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"`. Hàm truyền thẳng chuỗi Prompt đó vào mô hình (thông qua `llm_fn`) để sinh câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

========================================= 42 passed in 0.18s =========================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A                              | Câu B                                | Dự đoán | Điểm thực tế | Đúng? |
| ---- | ----------------------------------- | ------------------------------------- | ---------- | ---------------- | ------- |
| 1    | "Thời gian hoàn tiền"            | "Bao lâu nhận được tiền hoàn"  | cao        | 0.85             | Đúng  |
| 2    | "Điều kiện đổi trả"           | "Trường hợp được hoàn hàng"   | cao        | 0.89             | Đúng  |
| 3    | "Shopee Mall"                       | "Cửa hàng bán lẻ bên ngoài"     | thấp      | 0.23             | Đúng  |
| 4    | "Chế tài vi phạm"                | "Các loại hàng cấm"               | thấp      | 0.31             | Đúng  |
| 5    | "Người mua phải chịu phí ship" | "Người bán trả phí vận chuyển" | thấp      | 0.81             | Sai     |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> Bất ngờ nhất là Câu 5. Mặc dù nghĩa trái ngược (Người mua vs Người bán trả phí), nhưng điểm tương tự vẫn RẤT CAO. Lý do là Embedding Models tập trung vào việc 2 câu dùng chung không gian từ vựng/ngữ cảnh (phí vận chuyển, người dùng) hơn là logic suy luận "Ai trả".

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query)                                | Top-1 Chunk truy xuất được (tóm tắt)                   | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt)    |
| - | ------------------------------------------------ | ------------------------------------------------------------ | ------------ | --------------------------------- | ---------------------------------------- |
| 1 | Người Mua có bao nhiêu ngày hoàn hàng? | "Online - Người Bán khi đăng bán sách..." (Nhầm policy) | 0.3122 | KHÔNG | [Agent bị ngáo do Mock Embeddings sai] |
| 2 | Trường hợp nào được hoàn trả? | "ng quá trình giao dịch trên Sàn Shopee..." | 0.2864 | CÓ | Không nhận được Sản Phẩm / không nhận đủ... |
| 3 | Quy trình giải quyết tranh chấp? | "yết tranh chấp giữa Người Bán và Người Mua..." | 0.2789 | KHÔNG | [Agent bị ngáo do Mock Embeddings sai] |
| 4 | Chế tài khi bán hàng cấm? | "anh thu đơn hàng của Người Bán..." (Nhầm policy) | 0.2240 | KHÔNG | [Agent bị ngáo do Mock Embeddings sai] |
| 5 | Xử lý khiếu nại trong bao lâu? (lọc buyer) | "thể khiếu nại việc này lên các Cơ quan..." (Nhầm policy) | 0.2373 | KHÔNG | [Agent bị ngáo do Mock Embeddings sai] |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 1 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Chiến lược `FixedSizeChunker` của tôi tuy dễ cài đặt nhưng lại cắt gãy quá nhiều ngữ cảnh (đứt câu, mất tiêu đề). Khi so với thành viên dùng `RecursiveChunker`, kết quả của họ (3/5) tốt hơn hẳn so với của tôi (1/5) do giữ được trọn vẹn ngữ nghĩa của mỗi Mục luật.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                           | Điểm tự đánh giá |
| ---------------------------------------------------- | ---------------------- |
| Khởi động (Warm-up)                               | 5 / 5                  |
| Hướng tiếp cận của tôi (My Approach)           | 10 / 10                |
| Hoàn thiện code (Core Implementation — tests)     | 30 / 30                |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5                  |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10                 |
| **Tổng phần cá nhân**                      | **59 / 60**      |
