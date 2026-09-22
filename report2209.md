# Báo cáo thiết lập so sánh MPaGE, NSGA-II và MOEA/D trên bi-TSP20

## 1. Mục tiêu

Báo cáo này ghi lại các thay đổi đã thực hiện nhằm:

1. Sửa lỗi evaluator bi-TSP20 của MPaGE chấp nhận candidate không hợp lệ.
2. Đồng bộ dữ liệu và ngân sách đánh giá với các kết quả NSGA-II và MOEA/D đã có, để không phải chạy lại hai baseline.
3. Đưa các thành phần cốt lõi của MPaGE gần với thiết lập trong bài báo gốc.
4. Làm rõ mức độ công bằng và các giới hạn còn tồn tại khi so sánh hypervolume.

Hai file baseline đang được sử dụng là:

- `reproduce bitsp20/results/nsga_bi_tsp20.json`
- `reproduce bitsp20/results/moead_bi_tsp20.json`

`nsgaii.py` chỉ là entrypoint tương thích gọi implementation trong `nsga.py`.

## 2. Cấu hình baseline đã có

Các JSON hiện tại của NSGA-II và MOEA/D sử dụng cùng cấu hình:

| Tham số | Giá trị |
|---|---:|
| Bài toán | bi-TSP20 |
| Số instance | 4 |
| Data seed | 2025 |
| Algorithm seed theo instance | 2025, 2026, 2027, 2028 |
| Population tour | 100 |
| Generations | 20 |
| Offspring evaluations | 2.000/instance |
| Tổng objective evaluations | 2.100/instance |
| Reference point | `[20.0, 20.0]` |

Mean HV đã lưu:

- NSGA-II: `208.979934`.
- MOEA/D: `210.099747`.

Các baseline này không được chạy lại trong lần cấu hình MPaGE hiện tại.

## 3. Sửa evaluator MPaGE để chỉ đánh giá tour hợp lệ

### 3.1. Lỗi trong evaluator gốc

Heuristic lịch sử có phép perturbation tạo node ID dạng số thực:

```python
perturbation = random.uniform(-0.5, 0.5)
neighbor_solution[idx] = (neighbor_solution[idx] + perturbation) % n
```

`check_constraint` gốc kiểm tra độ dài, tính duy nhất của các giá trị float và miền `[0,n)`. Sau đó `tour_cost` lại ép mỗi node sang `int`. Hai số khác nhau như `5.4` và `5.8` vì thế cùng trở thành node `5` khi tính cost. Candidate có thể lặp thành phố, bỏ sót thành phố khác nhưng vẫn được đưa vào archive và dùng để tính HV.

### 3.2. Constraint đã được sửa

Hàm `check_constraint` trong `llm4ad/task/optimization/bi_tsp_semo/evaluation.py` hiện:

1. Ép candidate sang mảng số nguyên.
2. Yêu cầu candidate là mảng một chiều.
3. Yêu cầu đúng độ dài `problem_size`.
4. Yêu cầu tập node sau ép kiểu chính xác là hoán vị `0..problem_size-1`.

Candidate vượt qua kiểm tra được chuẩn hóa thành mảng nguyên trước khi tính objective và trước khi đưa vào archive. Vì vậy representation dùng để kiểm tra, tính cost và lưu archive là nhất quán.

Thay đổi này loại bỏ nguyên nhân làm HV của evaluator loose tăng giả tạo. Nó cũng có nghĩa kết quả chạy mới không còn tái lập nguyên xi HV được báo cáo bởi implementation gốc có lỗi.

## 4. Đồng bộ MPaGE với các baseline hiện có

### 4.1. Dữ liệu và seed

Evaluator MPaGE đã được đưa từ 10 về 4 instance. `GetData` vẫn dùng data seed `2025`, nên bốn instance đầu giống bốn instance đã dùng cho NSGA-II và MOEA/D.

Trước mỗi instance, evaluator đặt:

```text
instance 0 -> seed 2025
instance 1 -> seed 2026
instance 2 -> seed 2027
instance 3 -> seed 2028
```

Các giá trị seed trùng với baseline đã lưu. Tuy nhiên, cùng seed số học không đồng nghĩa quần thể khởi tạo giống từng tour, vì NSGA-II/MOEA-D và SEMO sử dụng các API sinh số ngẫu nhiên và toán tử khác nhau.

### 4.2. Ngân sách đánh giá một heuristic

Mỗi heuristic do MPaGE tạo ra được đánh giá với:

- 100 tour khởi tạo/instance.
- 2.000 lần thử sinh offspring/instance.
- 4 instance.
- Reference point `[20.0,20.0]`.

Do đó một heuristic MPaGE có:

```text
2.100 candidate attempts/instance
8.400 candidate attempts trên 4 instance
```

Con số này bằng ngân sách của một lần chạy NSGA-II hoặc MOEA/D trong các JSON hiện có nếu ngân sách được tính theo số candidate attempts.

### 4.3. Hai loại population không được nhầm lẫn

MPaGE có hai tầng population:

- **Population tour bên trong SEMO:** 100. Đây là đại lượng tương ứng với population 100 của NSGA-II/MOEA-D.
- **Population heuristic bên ngoài MPaGE:** 6. Đây là số chương trình heuristic được duy trì trong quá trình LLM thiết kế thuật toán, không tương ứng với population tour của baseline.

Việc giảm population heuristic từ 10 xuống 6 không làm population của bài toán TSP giảm từ 100 xuống 6.

## 5. Các tham số được đưa gần thiết lập bài báo

Bài báo gốc mô tả thiết lập:

- GPT-4o-mini, temperature `0.7`, để sinh heuristic.
- GPT-4o để assessment và semantic clustering.
- Population heuristic 10.
- 20 generations.
- Mỗi crossover chọn hai parent.
- 10 instance cho mỗi bài toán.
- SEMO chạy 2.000 iterations.
- Timeout 60 giây.
- `epsilon = 0.9`.
- `gamma = 0.3`.
- 4 PFG segments.

Các thay đổi trong code:

### 5.1. Model và API

- `main.py` dùng GPT-4o-mini với `temperature=0.7` cho generator.
- GPT-4o được dùng thật sự cho cả assessment/reflection và clustering.
- Adapter clustering trả structured output khi prompt yêu cầu trường `Group`, và trả text khi thực hiện assessment.
- Cờ `llm_review=True` được bật.
- Một API key dùng chung cho hai model; ưu tiên biến môi trường `OPENAI_API_KEY`, sau đó mới đọc `secret.txt`.
- `secret_cluster.txt` không còn được sử dụng.

### 5.2. PFG và variation

- PFG segments được cấu hình là 4.
- Xác suất chọn theo PFG được sửa từ mặc định `0.8` thành `epsilon=0.9`.
- Xác suất mutation `gamma=0.3` được bổ sung vào luồng sinh offspring.
- Với xác suất còn lại, thuật toán dùng crossover giữa hai parent.
- GPT-4o thực hiện semantic clustering và assessment trước khi GPT-4o-mini sinh offspring.

### 5.3. So sánh paper và cấu hình chạy hiện tại

| Tham số | Paper | Hiện tại | Trạng thái |
|---|---:|---:|---|
| Generator | GPT-4o-mini | GPT-4o-mini | Khớp |
| Generator temperature | 0.7 | 0.7 | Khớp |
| Assessor/clusterer | GPT-4o | GPT-4o | Khớp |
| PFG segments | 4 | 4 | Khớp |
| `epsilon` | 0.9 | 0.9 | Khớp |
| `gamma` | 0.3 | 0.3 | Khớp |
| Parent/crossover | 2 | 2 | Khớp |
| SEMO iterations | 2.000 | 2.000 | Khớp |
| Evaluation timeout | 60 giây | 60 giây | Khớp |
| Instance đánh giá heuristic | 10 | 4 | Cố ý giảm để khớp baseline đã có |
| Population heuristic MPaGE | 10 | 6 | Cố ý giảm chi phí API |
| Generations MPaGE | 20 | 10 | Cố ý giảm chi phí API |
| Số heuristic tối đa | khoảng 200 | 60 | Cố ý giảm chi phí API |

Như vậy model, vai trò model, PFG, xác suất variation, SEMO budget và timeout đã sát paper. Ba tham số về quy mô outer search được cố ý giảm; cấu hình hiện tại là cấu hình so sánh tiết kiệm, không phải bản tái lập đầy đủ chi phí của paper.

## 6. Số lần đánh giá candidate

| Phạm vi | NSGA-II | MOEA/D | MPaGE |
|---|---:|---:|---:|
| Tour attempts/instance | 2.100 | 2.100 | 2.100 cho mỗi heuristic |
| Tour attempts/4 instance | 8.400 | 8.400 | 8.400 cho mỗi heuristic |
| Số heuristic được thiết kế | Không áp dụng | Không áp dụng | Tối đa 60 |
| Tổng tour attempts của outer search | 8.400 | 8.400 | Tối đa 504.000 |

Tổng `504.000` được tính bằng:

```text
60 heuristic x 4 instance x 2.100 candidate attempts
```

Đây là cận theo số lần thử candidate. Số objective evaluations thực tế của MPaGE có thể thấp hơn vì candidate không hợp lệ bị loại trước khi gọi `tour_cost`. Số API requests cũng có thể lớn hơn 60 nếu LLM trả output không parse được, vì `max_sample_nums=60` đếm các heuristic đã đi tới bước đánh giá.

## 7. Đánh giá tính công bằng hiện tại

### 7.1. Những phần đã công bằng hơn

- Cùng bài toán bi-TSP20.
- Cùng bốn instance được tạo từ data seed `2025`.
- Cùng danh sách algorithm seed `2025..2028`.
- Cùng reference point `[20.0,20.0]`.
- Cùng 100 tour khởi tạo và 2.000 offspring attempts/instance khi xét một heuristic MPaGE.
- Mọi candidate được dùng để tính objective trong evaluator strict phải tương ứng với một hoán vị TSP hợp lệ sau chuẩn hóa.
- HV được tính riêng trên tập objective không bị thống trị cuối cùng của từng thuật toán.

### 7.2. Những phần vẫn chưa hoàn toàn công bằng

#### Tổng chi phí thiết kế

NSGA-II và MOEA/D chạy trực tiếp một lần với 8.400 objective evaluations. MPaGE dùng tối đa 60 lần đánh giá heuristic, tương ứng tối đa 504.000 tour attempts, chưa kể chi phí GPT-4o-mini và GPT-4o. Nếu chỉ so sánh runtime chạy heuristic cuối, chi phí thiết kế này bị loại khỏi phép so sánh.

Do đó cần công bố rõ hai loại chi phí:

1. **Design cost:** toàn bộ API calls và các lần đánh giá 60 heuristic.
2. **Deployment cost:** chi phí chạy một heuristic cuối trên bốn instance.

#### Candidate attempts và objective evaluations

NSGA-II và MOEA/D đánh giá objective cho đủ 2.100 tour/instance. SEMO strict vẫn thử 2.100 candidate/instance, nhưng offspring sai bị loại trước khi tính objective. Vì vậy số objective evaluations hợp lệ của MPaGE có thể thấp hơn baseline.

Thiết lập hiện tại công bằng theo **offspring attempts**, chưa công bằng tuyệt đối theo **objective evaluations hợp lệ**. Nếu muốn cân bằng theo objective evaluations, evaluator phải tiếp tục sinh cho tới khi đủ 2.000 offspring hợp lệ hoặc toán tử phải được sửa để luôn sinh hoán vị.

#### Cơ chế archive

MOEA/D và SEMO giữ external archive, trong khi NSGA-II hiện tính HV từ Pareto front của quần thể cuối. Một nghiệm tốt từng được NSGA-II tìm thấy nhưng đã bị loại khỏi quần thể sẽ không đóng góp vào HV. So sánh chặt hơn nên duy trì cùng một external nondominated archive cho cả ba thuật toán.

#### Initial population

Các thuật toán dùng cùng seed số học nhưng không dùng cùng bộ tour khởi tạo từng phần tử. Muốn paired comparison hoàn toàn, cần sinh trước một tập 100 tour cho mỗi instance và cấp đúng tập đó cho cả NSGA-II, MOEA/D và SEMO.

#### Meta-optimization trên chính tập đánh giá

MPaGE dùng bốn instance để chấm điểm và lựa chọn heuristic trong outer search. Nếu heuristic tốt nhất sau đó lại được báo cáo trên chính bốn instance này, kết quả là in-sample và có nguy cơ overfit. NSGA-II/MOEA-D không trải qua bước chọn chương trình trên bộ instance đó.

Để đánh giá khả năng tổng quát công bằng, cần tách:

- Training instances dùng để MPaGE thiết kế heuristic.
- Test instances chưa xuất hiện trong outer search, dùng chung để đánh giá heuristic cuối, NSGA-II và MOEA/D.

Việc này sẽ yêu cầu tạo hoặc chạy lại baseline trên test set mới. Nếu không muốn chạy lại baseline, kết quả hiện tại chỉ nên được mô tả là so sánh in-sample trên bốn instance đã có.

#### Số lần chạy ngẫu nhiên

Các JSON baseline hiện tại chứa một lần chạy cho mỗi instance. Bốn instance không tương đương với nhiều lần chạy ngẫu nhiên trên cùng một instance. Vì chưa chạy lại baseline, chưa thể đưa ra phân bố HV nhiều seed hoàn toàn paired cho MPaGE mới.

## 8. Vai trò của SEMO-loose

`semo_loose.py` chỉ dùng để tái lập và phân tích lỗi evaluator MPaGE gốc. Các Pareto front loose có thể chứa đường đi lặp hoặc thiếu thành phố sau ép kiểu. Vì vậy:

- Không dùng SEMO-loose để xếp hạng chính thức với NSGA-II/MOEA-D.
- Chỉ dùng nó để minh họa mức HV bị thổi phồng do constraint sai.
- Kết quả chính thức của lần chạy MPaGE mới phải dùng evaluator strict trong `llm4ad/task/optimization/bi_tsp_semo/evaluation.py`.

## 9. Quy trình chạy và so sánh hiện tại

### 9.1. Chạy MPaGE

Từ thư mục `MPaGE`:

```powershell
python main.py
```

API key được lấy theo thứ tự:

1. Biến môi trường `OPENAI_API_KEY`.
2. File `MPaGE/secret.txt`.

Mỗi lần chạy tạo thư mục timestamp mới dưới `MPaGE/logs/` và không ghi đè log cũ.

### 9.2. So sánh với baseline đã lưu

Sau khi MPaGE hoàn tất, cần chọn heuristic tốt nhất từ log mới và đánh giá nó trên đúng bốn instance/configuration nói trên. Có thể dùng `reproduce bitsp20/compare_hv.py` với đường dẫn log mới và hai JSON baseline hiện có.

Không nên dùng kết quả cũ của `semo_strict.py` để đại diện cho heuristic mới nếu MPaGE vừa sinh ra chương trình khác.

Khi báo cáo kết quả cuối, tối thiểu cần ghi:

- HV theo từng instance.
- Mean HV trên bốn instance.
- Số offspring bị constraint loại.
- Số objective evaluations thực tế.
- Runtime chạy heuristic cuối.
- Số heuristic/API calls của outer search.

## 10. Kết luận

Cấu hình hiện tại đã cải thiện tính công bằng ở mức dữ liệu, seed, reference point và ngân sách candidate attempts của một heuristic. Các thành phần model, assessment/clustering, PFG và xác suất variation cũng đã được đưa gần thiết lập paper.

Tuy nhiên, đây chưa phải phép so sánh hoàn toàn công bằng về tổng compute, objective evaluations, archive, initial population và train/test separation. Cách diễn giải phù hợp nhất hiện tại là:

> So sánh in-sample giữa heuristic tốt nhất do một MPaGE outer search rút gọn tạo ra và các kết quả NSGA-II/MOEA-D đã lưu, trên cùng bốn instance và cùng ngân sách 2.000 offspring attempts cho mỗi lần chạy heuristic.

Không nên diễn giải kết quả này như một phép tái lập đầy đủ paper hoặc một benchmark compute-matched hoàn toàn.
