# So sánh NSGA-II, SEMO và MOEA/D cho bi-TSP20

## Phạm vi thực hiện

- Thêm `nsga.py`: cài đặt NSGA-II cho biểu diễn hoán vị TSP, gồm non-dominated sorting, crowding distance, binary tournament, order crossover (OX), đột biến đảo đoạn/hoán đổi và environmental selection.
- Thêm `moead.py`: cài đặt MOEA/D với 100 vector trọng số đều trên hai mục tiêu, Tchebycheff decomposition, lân cận theo khoảng cách vector trọng số, giới hạn số nghiệm bị thay thế và external Pareto archive.
- Cập nhật `nsgaii.py` thành entrypoint tương thích, gọi trực tiếp chương trình trong `nsga.py`.
- `semo_strict.py`: chạy heuristic tốt nhất lịch sử của MPaGE, ép node ID sang số nguyên rồi kiểm tra candidate có phải hoán vị đầy đủ hay không.
- `semo_loose.py`: tái lập nguyên văn `select_neighbor`, `check_constraint` và `tour_cost` của MPaGE gốc.
- `compare_hv2.py`: chạy lại và so sánh HV của bốn thuật toán trên cùng dữ liệu, seed và reference point.
- `compare_pipeline.py`: lặp toàn bộ phép so sánh qua nhiều seed; mặc định là 10 lần.

## Thiết lập so sánh mặc định

Hai baseline dùng trực tiếp `GetData` của MPaGE, do đó chạy trên đúng dữ liệu mặc định:

- Bài toán: bi-objective TSP, 20 thành phố.
- Số instance: 4.
- Seed sinh dữ liệu: `2025` (được cố định trong `GetData`).
- Reference point tính hypervolume: `[20.0, 20.0]`.
- Population: 100; số thế hệ: 20.
- Ngân sách mỗi instance: 100 nghiệm khởi tạo + 2.000 lần thử sinh nghiệm con. Với NSGA-II, MOEA/D và SEMO-loose, cấu hình hiện tại dẫn đến 2.100 lần tính objective. SEMO-strict có thể ít hơn vì candidate không hợp lệ bị loại trước khi tính objective.
- Seed mặc định của thuật toán: `2025 + chỉ_số_instance`, giúp kết quả baseline tái lập được nhưng không làm các instance dùng cùng một chuỗi ngẫu nhiên.
- Khi chạy lại MPaGE, evaluator dùng đúng 4 instance, 100 tour ban đầu, 2.000 offspring attempts và các seed `2025..2028` để khớp các JSON baseline hiện có. Population heuristic bên ngoài của MPaGE là một ngân sách thiết kế khác, đã được giảm còn 6 heuristic × 10 generations, tối đa 60 lần sinh/đánh giá heuristic để giảm chi phí API; con số 6 này không thay thế population 100 của bài toán TSP.

Mỗi chương trình ghi một file JSON chứa cấu hình, runtime, hypervolume, kích thước Pareto front, objective vectors và tours của từng instance. Hypervolume dùng cùng `pymoo.indicators.hv.HV` như evaluator MPaGE.

## Cách chạy

Chạy từ thư mục `MPaGE/reproduce bitsp20`:

```bash
python nsga.py
python moead.py
python semo_strict.py
python semo_loose.py
python compare_hv2.py
python compare_pipeline.py
```

Kết quả mặc định:

- `results/nsga_bi_tsp20.json`
- `results/moead_bi_tsp20.json`
- `results/semo_bi_tsp20.json`
- `results/semo_loose_bi_tsp20.json`
- `results/hv_comparison_strict.json`
- `results/hv_comparison_strict.csv`
- `results/hv_pipeline_10_runs.json`
- `results/hv_pipeline_10_runs.csv`

Có thể xem tất cả tham số bằng `python nsga.py --help` và `python moead.py --help`. Ví dụ đổi seed và file đầu ra:

```bash
python nsga.py --seed 42 --output results/nsga_seed42.json
python moead.py --seed 42 --output results/moead_seed42.json
```

Để chạy pipeline 10 seed mặc định:

```bash
python compare_pipeline.py
```

Có thể thay đổi số lần chạy, ví dụ `python compare_pipeline.py --runs 20`. Runtime của MPaGE ở đây chỉ bao gồm thời gian chạy heuristic trong SEMO; thời gian sinh heuristic bằng LLM nằm ở tầng thiết kế thuật toán và không được đưa vào benchmark.

## Phân tích vấn đề trong thuật toán MPaGE gốc

### Hiện tượng: evaluator chấp nhận candidate không phải tour TSP hợp lệ

Một nghiệm hợp lệ của bi-TSP20 phải là hoán vị nguyên của `0..19`: mỗi thành phố xuất hiện đúng một lần. Tuy nhiên, heuristic tốt nhất được lưu trong `pop_18.json` có thể tạo node ID dạng số thực. Phần mã tương ứng nằm tại `semo_loose.py`, dòng 55–61:

```python
perturbation_indices = random.sample(range(n), k=min(3, n))
for idx in perturbation_indices:
    perturbation = random.uniform(-0.5, 0.5)
    neighbor_solution[idx] = (neighbor_solution[idx] + perturbation) % n

return np.array(neighbor_solution)
```

Phép cộng `perturbation` biến một tour nguyên thành mảng `float`. Phép `% n` chỉ giữ giá trị trong khoảng `[0,n)`, không bảo đảm đó là chỉ số thành phố nguyên và cũng không bảo đảm tour vẫn là hoán vị sau khi ép kiểu.

### Lỗi thứ nhất: `check_constraint` kiểm tra trên giá trị float

Hàm gốc nằm tại `llm4ad/task/optimization/bi_tsp_semo/evaluation.py`, dòng 54–62, và được sao chép nguyên văn vào `semo_loose.py`, dòng 99–107:

```python
def check_constraint(solution, problem_size):
    sol = list(solution)
    if len(sol) != problem_size:
        return False
    if len(set(sol)) != problem_size:
        return False
    if not all(0 <= x < problem_size for x in solution):
        return False
    return True
```

Logic này chỉ kiểm tra ba điều kiện:

1. Candidate có đúng 20 phần tử.
2. Các giá trị float khác nhau trước khi ép kiểu.
3. Các giá trị nằm trong `[0,20)`.

Hàm không kiểm tra dtype nguyên và không kiểm tra candidate có đúng tập `{0,1,...,19}` hay không. Do đó các giá trị như `5.4` và `5.8` được xem là hai node khác nhau và vẫn vượt qua constraint.

### Lỗi thứ hai: `tour_cost` lại ép node sang `int`

Sau khi candidate float đã vượt qua constraint, `tour_cost` tại `evaluation.py`, dòng 24–25 và 33, thực hiện:

```python
node1, node2 = int(solution[j]), int(solution[j + 1])
node_first, node_last = int(solution[0]), int(solution[-1])
```

Đây là sự không nhất quán cốt lõi: constraint đánh giá tính duy nhất trên số thực, nhưng objective lại đánh giá đường đi sau khi cắt phần thập phân. Ví dụ, `5.4` và `5.8` khác nhau khi chạy `set(solution)`, nhưng cả hai đều trở thành node `5` khi tính cost. Khi đó tour đi qua node 5 hai lần và bỏ sót ít nhất một thành phố khác.

### Lỗi thứ ba: candidate sai tiếp tục được đưa vào archive và tính HV

Vòng đánh giá gốc tại `evaluation.py`, dòng 76–87, chỉ gọi `check_constraint` lỏng trước khi tính objective:

```python
s_prime = eva(Archive, instance, distance_matrix_1, distance_matrix_2)
if not check_constraint(s_prime, problem_size):
    continue
f_s_prime = tour_cost(instance, s_prime, problem_size)
...
Archive.append((s_prime, f_s_prime))
```

Candidate float được lưu nguyên trạng trong archive. Sau đó evaluator lấy toàn bộ objective trong archive và tính HV tại dòng 89–94. Vì các đường đi sau ép kiểu có thể lặp node và bỏ node, cost của chúng thường thấp giả tạo so với một chu trình Hamilton hợp lệ. Đây là nguyên nhân HV của chương trình được đánh giá cao dù nghiệm không thỏa bài toán bi-TSP20.

### Cách làm chặt constraint

Phiên bản `semo_strict.py`, dòng 63–73, ép kiểu trước rồi kiểm tra kết quả có đúng hoán vị đầy đủ hay không:

```python
def check_constraint(solution: np.ndarray, problem_size: int) -> bool:
    try:
        candidate = np.asarray(solution, dtype=int)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        candidate.ndim == 1
        and len(candidate) == problem_size
        and np.array_equal(np.sort(candidate), np.arange(problem_size))
    )
```

Nếu candidate vượt qua kiểm tra, vòng lặp tại dòng 126–131 chuyển candidate sang mảng nguyên trước khi tính objective và trước khi đưa vào archive. Nhờ đó representation được kiểm tra, representation dùng để tính cost và representation lưu trong archive là cùng một tour.

### Kết quả trước và sau khi làm chặt constraint

Kết quả dưới đây dùng cùng bốn instance, base seed `2025`, 100 nghiệm ban đầu, 2.000 lần thử sinh offspring và reference point `[20.0,20.0]`:

| Instance | SEMO-loose HV | SEMO-strict HV | Candidate bị strict loại | Strict objective evaluations |
|---:|---:|---:|---:|---:|
| 0 | 252.027732 | 163.118844 | 1.754 | 346 |
| 1 | 276.002239 | 199.254963 | 1.758 | 342 |
| 2 | 281.478265 | 201.824634 | 1.750 | 350 |
| 3 | 268.129229 | 176.545627 | 1.757 | 343 |
| **Mean** | **269.409366** | **185.186017** | **1.755** | **345,25** |

Sau khi làm chặt constraint, mean HV giảm `84.223349`, tương đương khoảng `31,26%`. Trong lần chạy loose, số tour trên Pareto front của bốn instance lần lượt là 46, 10, 18 và 28; không tour nào trong số này trở thành hoán vị `0..19` hợp lệ sau khi ép sang `int`.

Kết quả lịch sử được ghi trong `pop_18.json` là mean HV `271.286452`; lần tái lập loose hiện tại đạt `269.409366`. Hai con số gần nhau và đều chịu ảnh hưởng của cùng lỗ hổng constraint. Khi constraint được sửa, mean HV còn `185.186017`, thấp hơn NSGA-II (`208.979934`) và MOEA/D (`210.099747`) trong cùng cấu hình.

### Diễn giải kết quả

Mức giảm HV cho thấy thành tích cao của heuristic trong evaluator gốc phụ thuộc đáng kể vào việc candidate không hợp lệ được chấp nhận và tính cost. Vì vậy không nên dùng HV loose để kết luận heuristic vượt trội trên bi-TSP20 hợp lệ.

Cũng cần lưu ý SEMO-strict hiện chỉ có khoảng 342–350 objective evaluations vì candidate sai bị loại, trong khi NSGA-II và MOEA/D có 2.100 evaluations. Kết quả strict chứng minh tác động của việc sửa constraint, nhưng để benchmark hiệu năng thuật toán hoàn toàn công bằng cần sửa toán tử `select_neighbor` để luôn sinh hoán vị hợp lệ, hoặc tiếp tục sinh cho đến khi đủ 2.000 offspring hợp lệ.

## Kiểm thử đã thực hiện

- Biên dịch cú pháp thành công cho toàn bộ các script baseline, SEMO và so sánh.
- Smoke test với population 10, 2 thế hệ xác nhận NSGA-II, MOEA/D và SEMO-strict chỉ đưa tour nguyên hợp lệ vào front; SEMO-loose tái lập đúng hành vi constraint gốc.
- Chạy hoàn chỉnh cấu hình mặc định thành công. Kết quả baseline được lưu trong thư mục `results/`:
  - NSGA-II: mean HV `208.979934`, mean runtime khoảng `2.972155` giây/instance.
  - MOEA/D: mean HV `210.099747`, mean runtime khoảng `0.496553` giây/instance.
- Kết quả một lần chạy so sánh với base seed `2025`:
  - NSGA-II: mean HV `208.979934`.
  - SEMO-strict: mean HV `185.186017`.
  - SEMO-loose: mean HV `269.409366`, nhưng front không gồm các hoán vị TSP hợp lệ sau ép kiểu.
  - MOEA/D: mean HV `210.099747`.

Các giá trị hypervolume tái lập được với cùng seed; runtime phụ thuộc máy và tải hệ thống nên chỉ nên xem các số trên là smoke benchmark của môi trường hiện tại.
