# Baseline NSGA-II và MOEA/D cho bi-TSP20

## Phạm vi thực hiện

- Thêm `nsga.py`: cài đặt NSGA-II cho biểu diễn hoán vị TSP, gồm non-dominated sorting, crowding distance, binary tournament, order crossover (OX), đột biến đảo đoạn/hoán đổi và environmental selection.
- Thêm `moead.py`: cài đặt MOEA/D với 100 vector trọng số đều trên hai mục tiêu, Tchebycheff decomposition, lân cận theo khoảng cách vector trọng số, giới hạn số nghiệm bị thay thế và external Pareto archive.
- Cập nhật `nsgaii.py` thành entrypoint tương thích, gọi trực tiếp chương trình trong `nsga.py`.

## Thiết lập so sánh mặc định

Hai baseline dùng trực tiếp `GetData` của MPaGE, do đó chạy trên đúng dữ liệu mặc định:

- Bài toán: bi-objective TSP, 20 thành phố.
- Số instance: 4.
- Seed sinh dữ liệu: `2025` (được cố định trong `GetData`).
- Reference point tính hypervolume: `[20.0, 20.0]`.
- Population: 100; số thế hệ: 20.
- Ngân sách mỗi instance: 100 nghiệm khởi tạo + 20 × 100 nghiệm con = 2.100 lần đánh giá tour. Con số này khớp với evaluator MPaGE: 100 nghiệm khởi tạo + 2.000 lần gọi heuristic.
- Seed mặc định của thuật toán: `2025 + chỉ_số_instance`, giúp kết quả baseline tái lập được nhưng không làm các instance dùng cùng một chuỗi ngẫu nhiên.

Mỗi chương trình ghi một file JSON chứa cấu hình, runtime, hypervolume, kích thước Pareto front, objective vectors và tours của từng instance. Hypervolume dùng cùng `pymoo.indicators.hv.HV` như evaluator MPaGE.

## Cách chạy

Chạy từ thư mục `MPaGE/reproduce bitsp20`:

```bash
python nsga.py
python moead.py
```

Kết quả mặc định:

- `results/nsga_bi_tsp20.json`
- `results/moead_bi_tsp20.json`

Có thể xem tất cả tham số bằng `python nsga.py --help` và `python moead.py --help`. Ví dụ đổi seed và file đầu ra:

```bash
python nsga.py --seed 42 --output results/nsga_seed42.json
python moead.py --seed 42 --output results/moead_seed42.json
```

Để so sánh thực nghiệm đáng tin cậy, nên chạy nhiều seed cho cả ba phương pháp và báo cáo trung bình cùng độ lệch chuẩn của hypervolume/runtime. Runtime của MPaGE bao gồm thời gian chạy heuristic trong SEMO, còn thời gian sinh heuristic bằng LLM nằm ở tầng thiết kế thuật toán; cần ghi rõ phạm vi runtime khi lập bảng so sánh.

## Kiểm thử đã thực hiện

- Biên dịch cú pháp thành công bằng `python -m py_compile nsga.py nsgaii.py moead.py`.
- Smoke test với population 10, 2 thế hệ xác nhận mỗi thuật toán dùng đúng 30 lần đánh giá, mọi tour đều là hoán vị hợp lệ của 20 thành phố và front trả về không chứa nghiệm bị thống trị.
- Chạy hoàn chỉnh cấu hình mặc định thành công; mỗi thuật toán dùng đúng 2.100 lần đánh giá trên từng instance. Kết quả của lần kiểm thử này đã được lưu trong thư mục `results/`:
  - NSGA-II: mean HV `208.979934`, mean runtime khoảng `2.972155` giây/instance.
  - MOEA/D: mean HV `210.099747`, mean runtime khoảng `0.496553` giây/instance.

Các giá trị hypervolume tái lập được với cùng seed; runtime phụ thuộc máy và tải hệ thống nên chỉ nên xem các số trên là smoke benchmark của môi trường hiện tại.
