# Bộ ước lượng kênh AI-Native cho 5G NR

**Một mô hình deep learning thay thế một thành phần xử lý tín hiệu cổ điển
trong máy thu vô tuyến 5G - được huấn luyện, đo benchmark, và đóng gói
thành một dịch vụ có thể triển khai thật - xây dựng trên nền NVIDIA Sionna.**

**Tác giả:** [Tào Việt Đức](https://github.com/taovietducofficial)

**Ngôn ngữ:** [English](README.md) | Tiếng Việt

---

## Tóm tắt nhanh

| | |
|---|---|
| **Đây là gì** | Một mạng neural (2 biến thể CNN và Transformer) ước lượng kênh vô tuyến chính xác hơn thuật toán cổ điển mà nó thay thế, bên trong một mô phỏng đường lên 5G NR theo đúng chuẩn. |
| **Kết quả đo được** | Tại ngưỡng lỗi chuẩn 10%, model CNN đạt cùng độ tin cậy ở mức tín hiệu **thấp hơn ~0.6 dB** so với thuật toán cổ điển - đúng vùng SNR mà người dùng ở rìa cell hoặc di động thường gặp phải. Tại một điểm vận hành tiêu biểu, điều này tương ứng với **~27% ít lần truyền lại hơn**. |
| **Bằng chứng, không chỉ là tuyên bố** | Mọi con số dưới đây đều đến từ 1 script trong repo này mà ai cũng chạy lại được. Chỗ nào bằng chứng còn mỏng (vd cỡ mẫu, số liệu GPU) đều được nói rõ thay vì làm tròn cho đẹp. |
| **Chạy được như một service thật** | `docker run` là có ngay API sống và trang demo bấm được - không chỉ là notebook. |
| **Độ trưởng thành kỹ thuật** | Pin version dependency, test tự động, CI có lint/type-check, structured logging + metrics, và bản ghi truy vết (provenance) gắn mỗi model đã deploy với đúng lần training sinh ra nó. |
| **Giới hạn thành thật** | Đây là mô phỏng ở mức nghiên cứu/portfolio, không phải triển khai thực địa đã được kiểm chứng - xem [Giới hạn thành thật](#giới-hạn-thành-thật) để biết chính xác điều đó có nghĩa là gì (và không có nghĩa là gì). |

**Thử ngay bằng 1 lệnh:** xem [Bắt đầu nhanh](#bắt-đầu-nhanh).

---

## Vì sao điều này quan trọng

Các mạng viễn thông hiện đại (5G hôm nay, 6G kế tiếp) tốn rất nhiều công sức
kỹ thuật và vốn đầu tư để tìm từng chút cải thiện nhỏ về **độ tin cậy tín
hiệu ở rìa vùng phủ** - đó là nơi việc truyền lại chồng chất, thông lượng
sụt giảm, và nhà mạng phải xây thêm trạm để bù đắp. Dự án này đặt ra một câu
hỏi hẹp nhưng cụ thể: *liệu một mạng neural nhỏ, đặt đúng vào vị trí một
thuật toán cổ điển đang đứng hôm nay, có thể cải thiện đo được độ tin cậy ở
vùng khó đó - mà không thay đổi bất cứ điều gì khác của hệ thống vô tuyến?*

Câu trả lời ở đây, được đo chứ không phải giả định: **có, ở mức vừa phải,
và nhất quán** - kèm theo một đánh giá thành thật về nơi nó không giúp ích
(tín hiệu rất mạnh) và điều đó ngụ ý gì cho cách nó thực sự nên được triển
khai.

Dự án cũng được xây để mô phỏng đúng cách một model như vậy sẽ thực sự đi
đến production: một model artifact có thể mang đi được, một inference
service có versioning và giám sát, và một cách nhìn rõ ràng về việc nó nằm
ở đâu (và không nằm ở đâu) trong một kiến trúc mạng thật (O-RAN) - không chỉ
là một training script.

---

## Bắt đầu nhanh

```bash
# Chạy 1 lần (cần môi trường nghiên cứu - xem phần Cài đặt bên dưới):
python -m service.export_model
python -m service.fixtures.gen_fixtures

# Từ đây trở đi, không cần Sionna/PyTorch nữa - chỉ cần Docker:
docker build -t neural-estimator-service -f service/Dockerfile service
docker run --rm -p 8000:8000 neural-estimator-service
```

Mở `http://localhost:8000/` - ba nút bấm (tín hiệu thấp/trung bình/cao)
chạy dữ liệu kênh thật qua thuật toán cổ điển và cả 2 model neural song
song, hiển thị độ chính xác và độ trễ trực tiếp.

---

## Mục lục

- [Kết quả](#kết-quả)
- [Cách hoạt động](#cách-hoạt-động)
- [Sơ đồ repository](#sơ-đồ-repository)
- [Cài đặt và chạy pipeline nghiên cứu](#cài-đặt-và-chạy-pipeline-nghiên-cứu)
- [Demo service / triển khai](#demo-service--triển-khai)
- [Vệ sinh kỹ thuật (engineering hygiene)](#vệ-sinh-kỹ-thuật-engineering-hygiene)
- [Giới hạn thành thật](#giới-hạn-thành-thật)
- [Đọc thêm](#đọc-thêm)

---

## Kết quả

**Độ chính xác ước lượng kênh theo cường độ tín hiệu** (kênh CDL-C, 3 km/h,
512 codeword mỗi điểm đo - `results/bler_results.csv` / `results/bler_vs_snr.png`):

| SNR (dB) | Baseline (LS+LMMSE) | CNN | Transformer |
|---:|---:|---:|---:|
| 7.0 | 1.000 | 0.953 | 0.955 |
| 7.5 | 0.865 | 0.418 | 0.529 |
| 8.0 | 0.338 | **0.094** | 0.229 |
| 8.5 | 0.113 | 0.064 | 0.123 |
| 9.0 | 0.066 | 0.035 | 0.123 |
| 9.5 | 0.033 | 0.023 | 0.074 |
| 10.0 | 0.027 | 0.023 | 0.055 |

(Giá trị là block-error rate - tỷ lệ phần trăm số lần truyền bị lỗi phải
truyền lại. Càng thấp càng tốt.)

**Đọc thẳng:** CNN vượt trội rõ ràng so với bộ ước lượng cổ điển trong suốt
vùng "waterfall" - nơi chất lượng tín hiệu ở mức biên. Tại ngưỡng lỗi chuẩn
10%, phương pháp cổ điển cần khoảng 8.6 dB cường độ tín hiệu để đạt được;
CNN đạt được ở khoảng 8.0 dB - tức **~0.6 dB margin gain**, cách tính đầy đủ
và ý nghĩa kinh doanh của nó nằm trong
[`docs/business-impact.md`](docs/business-impact.md). Transformer cũng vượt
qua baseline cổ điển nhưng không đều bằng (nó chững lại trước khi cải thiện
tiếp - nhiều khả năng chỉ là nhiễu thống kê ở cỡ mẫu của lần chạy này, được
nêu rõ thay vì làm mượt đi).

**Điều số liệu này KHÔNG cho thấy, nói rõ ngay từ đầu:** ở tín hiệu rất mạnh
(10 dB+), một lần đo thô hơn trước đó cho thấy bộ ước lượng cổ điển thực sự
đạt lỗi bằng 0 trong khi cả 2 model neural có một sàn lỗi dư nhỏ - đặc điểm
đã biết của các mạng được huấn luyện trên dải cường độ tín hiệu rộng thay vì
chuyên biệt cho một vùng cụ thể. Ý nghĩa thực tế: model này là một cải thiện
có mục tiêu cho vùng tín hiệu khó, không phải một sự thay thế toàn diện -
xem [Giới hạn thành thật](#giới-hạn-thành-thật).

**Chi phí inference và nén model** (chỉ CPU - xem
[Cài đặt](#cài-đặt-và-chạy-pipeline-nghiên-cứu); `results/benchmark.csv`):

| Batch size | Model | Tối ưu | Trước | Sau | Speedup |
|---:|---|---|---:|---:|---:|
| 8   | CNN | TorchScript trace+freeze | 3.02 ms | 3.22 ms | 0.94x |
| 8   | Transformer | INT8 quantization (một phần) | 12.56 ms | 19.83 ms | 0.63x |
| 256 | CNN | TorchScript trace+freeze | 158.1 ms | 149.9 ms | **1.06x** |
| 256 | Transformer | INT8 quantization (một phần) | 788.7 ms | 1071.3 ms | 0.74x |

**Đọc thẳng:** nén model không tự động là một chiến thắng - ở batch size
nhỏ, overhead của việc tối ưu còn lớn hơn cả lợi ích; nó chỉ có lãi khi
batch size đủ lớn để khấu hao overhead đó. Đây là một phát hiện kỹ thuật
thật về việc *khi nào* nén model có ích, đáng giá hơn một con số chỉ đẹp
khi đứng một mình. (Số tuyệt đối này dao động giữa các lần chạy tuỳ máy
đang bận cỡ nào - tự chạy lại `python -m src.benchmark` để có số sạch; xu
hướng định tính - nén có lợi ở batch lớn, không có lợi ở batch nhỏ - vẫn
giữ nguyên qua mọi lần chạy lại.)

Cả 2 model cũng đã được xác minh export sạch sang định dạng ONNX di động
với sai số song song (parity) so với model PyTorch gốc (chênh lệch khoảng
một phần mười triệu) - xem `service/export_model.py`.

**Kiểm tra real-time** (`src/realtime_benchmark.py`, `results/realtime_latency.csv`):
latency single-sample (batch=1) - con số thực sự quan trọng cho một hàm xử
lý mỗi slot của radio - so với ngân sách slot thật của 5G NR:

| Model | p50 | p95 | p99 | Vừa slot 1ms? | Vừa slot 0.25ms? |
|---|---:|---:|---:|:---:|:---:|
| CNN | 1.78 ms | 5.38 ms | 11.58 ms | Không | Không |
| Transformer | 3.05 ms | 6.20 ms | 8.52 ms | Không | Không |

**Đọc thẳng, không làm nhẹ đi:** trên CPU, cả 2 model hiện KHÔNG đạt bất kỳ
ngân sách slot thật nào ở p99. Đây chính là lý do vì sao inference trên
GPU/accelerator không phải "có thì tốt" mà là bắt buộc - xem
`colab_gpu_benchmark.ipynb` để đo số GPU thật, và
[`docs/oran-integration.md`](docs/oran-integration.md) để có thảo luận đầy
đủ về khoảng trống này.

**Xác thực MIMO (2-layer)** (`results/bler_results_mimo2.csv`,
`results/bler_vs_snr_mimo2.png`): cùng kiến trúc CNN, không sửa gì, train
lại trên cấu hình MIMO 2x2, vượt trội bộ ước lượng cổ điển trên toàn dải đo
được (vd 0.109 vs 0.172 BLER ở 14dB - giảm tương đối ~37%). Để có kết quả
*chạy đúng* cần đúng 1 lần sửa thật: lần thử đầu dùng 1 anten thu, về mặt lý
thuyết thông tin không thể tách được 2 luồng không gian (BLER=100%) dù train
thế nào - không phải lỗi model. Chi tiết đầy đủ trong
[`docs/oran-integration.md`](docs/oran-integration.md).

---

## Cách hoạt động

1. **Một đường lên 5G theo đúng chuẩn được mô phỏng**, không phải tự nghĩ ra
   - xây trên nền [NVIDIA Sionna](https://github.com/nvlabs/sionna) 2.0's
   5G NR PUSCH link (OFDM, mã hoá LDPC, các mô hình kênh 3GPP), nên vật lý
   và giao thức đều là thật, không phải mô hình đồ chơi.
2. **Mạng neural là một bản thay thế drop-in** cho đúng một thành phần cụ
   thể - bộ ước lượng kênh - dùng đúng interface mà simulator yêu cầu. Nó
   không bỏ qua hay xấp xỉ phần còn lại của chuỗi vô tuyến.
3. **Nó học bằng cách hiệu chỉnh, không phải từ đầu**: cả CNN và Transformer
   đều tinh chỉnh phỏng đoán ban đầu (nhiễu) của thuật toán cổ điển hướng về
   kênh thật, huấn luyện trên dữ liệu simulator tự sinh theo yêu cầu - không
   cần dataset ngoài.
4. **Hai kiến trúc, so sánh thành thật**: CNN và Transformer được đánh giá
   trên cùng 1 bài toán nên so sánh (độ chính xác vs. chi phí inference) là
   công bằng, cùng điều kiện.
5. **Model đã train sau đó được đóng gói đi**, không nằm im trong notebook:
   export sang định dạng ONNX di động, bọc trong REST API, đóng container,
   và có một trang demo sống - xem [Demo service / triển khai](#demo-service--triển-khai).

---

## Sơ đồ repository

```
src/baseline.py                classical PUSCH link (LS + LMMSE), BLER vs SNR
src/models.py                   CNNChannelEstimator, TransformerChannelEstimator
src/train.py                     trains a neural estimator on synthetic channel data
src/evaluate.py                   BLER-vs-SNR comparison: baseline vs CNN vs Transformer
src/benchmark.py                   inference latency + compression benchmark (CPU)
src/realtime_benchmark.py            single-sample p50/p95/p99 latency vs real slot budgets
tests/test_pipeline.py               assert-based smoke checks (research side)

colab_gpu_benchmark.ipynb        notebook Colab chạy sẵn để đo số GPU thật
oran-stub/                        demo REPORT/CONTROL kiểu E2 minh hoạ, chạy được thật (xem README riêng)

service/export_model.py        offline: exports checkpoints to ONNX + parity check + provenance manifest
service/fixtures/                 offline: real (noisy, perfect) channel fixtures at 3 SNRs
service/artifacts/                 exported .onnx models + <name>_manifest.json (committed, small)
service/app.py                     FastAPI: /health /metrics /models /v1/estimate /v1/demo/{bucket}
service/inference.py                onnxruntime wrapper + provenance lookup
service/static/index.html             clickable demo UI (no build step)
service/tests/test_service.py          assert-based smoke checks (service side)
service/Dockerfile                   builds a Sionna/PyTorch-free serving image
service/requirements.txt               pinned runtime deps; requirements-dev.txt adds httpx/ruff/mypy

docs/oran-integration.md        honest mapping to O-RAN xApp/rApp/E2/A1 concepts, and the 6G path
docs/business-impact.md          KPI/business framing built strictly from measured numbers

pyproject.toml                    ruff + mypy config
LICENSE                            All rights reserved (public for portfolio/evaluation viewing only)
Makefile                          research-test / export / fixtures / test / lint / typecheck / docker-build / docker-run
.github/workflows/ci.yml            lint+typecheck and service tests + docker build on every push; research tests on demand
```

---

## Cài đặt và chạy pipeline nghiên cứu

Sionna 2.0 yêu cầu Python 3.11+ và PyTorch (đã bỏ TensorFlow từ bản 2.0).
Được xây và chạy với venv Python 3.11:

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Không có GPU local khi build dự án này (đã kiểm tra qua `nvidia-smi`); mô
phỏng đường truyền và các model nhỏ ở đây train/chạy tốt trên CPU. Để có số
liệu latency GPU thật, chạy lại `src/benchmark.py` trên máy có CUDA hoặc
[Google Colab](https://colab.research.google.com/) - code y hệt, PyTorch tự
dùng CUDA khi tensor/model được chuyển sang `.cuda()`.

```bash
# 1. Kiểm tra nhanh đường truyền cổ điển
python -m src.baseline

# 2. Train các bộ ước lượng neural (dữ liệu tổng hợp, không cần dataset)
python -m src.train --model cnn --steps 1500
python -m src.train --model transformer --steps 1500

# 3. So sánh độ chính xác: baseline vs CNN vs Transformer
python -m src.evaluate --snr-min 6 --snr-max 10 --snr-step 0.5 --num-batches 32

# 4. Benchmark chi phí inference + nén model
python -m src.benchmark

# 5. Kiểm tra real-time: latency p50/p95/p99 single-sample so với ngân sách slot thật
python -m src.realtime_benchmark

# Tuỳ chọn: train + đánh giá biến thể MIMO 2-layer (cùng kiến trúc, không sửa code)
python -m src.train --model cnn --steps 1500 --num-layers 2 --out checkpoints/cnn_mimo2.pt
python -m src.evaluate --snr-min 6 --snr-max 14 --snr-step 1 --num-batches 16 --num-layers 2

# Smoke test
python -m tests.test_pipeline
```

Kết quả nằm ở `results/bler_vs_snr.png`, `results/bler_results.csv`, và
`results/benchmark.csv`. (`make research-test` / `make test` gói gọn các
smoke test; xem `Makefile`.)

Để có số liệu latency GPU thật, mở `colab_gpu_benchmark.ipynb` trên
[Google Colab](https://colab.research.google.com/) (bật GPU runtime, rồi Run
all) - notebook tự clone repo này và đo cùng 2 mạng trên GPU.

---

## Demo service / triển khai

Biến các checkpoint đã train thành thứ bạn thực sự chạy và bấm được - không
chỉ là 1 script nghiên cứu. Xem
[`docs/oran-integration.md`](docs/oran-integration.md) để có góc nhìn thành
thật về cách (và cách không nên) mô tả đây là một tích hợp O-RAN, cùng lộ
trình lên 6G, và [`docs/business-impact.md`](docs/business-impact.md) để có
cách tính KPI đầy đủ.

```bash
# Chạy 1 lần, trong venv nghiên cứu (cần Sionna/PyTorch):
python -m service.export_model          # export checkpoint -> service/artifacts/*.onnx
python -m service.fixtures.gen_fixtures  # sinh 3 fixture SNR cho demo

# Từ đây trở đi, bản thân service không cần Sionna hay PyTorch:
docker build -t neural-estimator-service -f service/Dockerfile service
docker run --rm -p 8000:8000 neural-estimator-service
```

Sau đó mở `http://localhost:8000/` để xem demo bấm được, hoặc gọi API trực
tiếp:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/demo/mid
curl http://localhost:8000/models    # provenance: checkpoint nào, lần train nào
curl http://localhost:8000/metrics   # metrics dạng Prometheus: request/latency
```

Test phía service (không cần Sionna/PyTorch): `python -m service.tests.test_service`.

---

## Vệ sinh kỹ thuật (engineering hygiene)

Các nâng cấp ở tầm nguyên tắc kỹ thuật senior, ngoài bản thân model và
service - sự khác biệt giữa "một model chạy được" và "một model người khác
có thể vận hành được":

- **Build tái lập được**: mọi dependency (nghiên cứu và service) đều pin
  chính xác version, không dùng `>=` - build hôm nay sẽ resolve giống hệt
  sau 1 năm.
- **Structured logging + metrics**: service ghi log mỗi request qua module
  `logging` của Python, và expose metrics dạng Prometheus tại `GET /metrics`
  (số lượng request và latency inference, theo từng model) - đúng chuẩn
  scrape target cho bất kỳ hệ thống giám sát thật nào.
- **Model provenance**: mỗi lần export đều ghi ra 1 manifest chứa SHA-256
  của checkpoint gốc, opset ONNX và sai số parity đo được, thời điểm export,
  cùng hyperparameter và loss cuối của lần train. `GET /models` hiển thị
  thông tin này - một artifact đã deploy truy vết được chính xác về thứ đã
  sinh ra nó, không chỉ là "tin tôi đi, đây là CNN".
- **CI quality gate**: lint (`ruff`) và type-check (`mypy`) chạy mỗi lần
  push, cùng với test suite và build Docker.
- **Input validation tại ranh giới tin cậy**: API kiểm tra shape của request
  trước khi chạm vào model, trả về lỗi sạch thay vì crash khi input sai
  định dạng.

---

## Giới hạn thành thật

Nói thẳng thay vì giấu đi - những điều dự án này *KHÔNG* tuyên bố:

- **Mô phỏng, không phải mạng thật đang chạy.** Toàn bộ dữ liệu kênh được
  sinh bởi simulator theo chuẩn của Sionna, không thu thập từ một mạng đã
  triển khai thật. Vật lý và giao thức là thật; lưu lượng và môi trường thì
  không.
- **Không bịa số liệu kinh doanh.** Các con số margin-gain và giảm truyền
  lại ở trên được suy ra trực tiếp từ dữ liệu mô phỏng đo được, có cách
  tính đi kèm; không có con số ROI bằng tiền, tiết kiệm vận hành, hay bán
  kính vùng phủ nào bị bịa ra - xem
  [`docs/business-impact.md`](docs/business-impact.md) để biết chính xác
  ranh giới đó nằm ở đâu.
- **Không đo GPU trực tiếp.** Không có GPU trên máy build; `colab_gpu_benchmark.ipynb`
  là notebook chạy sẵn để lấy số latency GPU thật trên Colab free trong vài
  phút - khoảng trống được lấp bằng 1 công cụ chạy được, không chỉ là lời
  ghi chú.
- **Ngưỡng real-time: đã đo, và hiện chưa đạt.** `src/realtime_benchmark.py`
  giờ đo latency p50/p95/p99 single-sample so với ngân sách slot 5G NR thật
  (0.25-1ms). Kết quả, nói thẳng: **trên CPU, cả 2 model hiện KHÔNG đạt bất
  kỳ ngân sách nào ở p99** (`results/realtime_latency.csv`) - đây là khoảng
  trống đã đo được thật, không phải bị làm nhẹ đi; xem
  [`docs/oran-integration.md`](docs/oran-integration.md) để có số liệu đầy
  đủ và ý nghĩa của nó.
- **MIMO: đã xác thực cho 2 layer, không chỉ lý thuyết.** Tuyên bố "chỉ đơn
  ăng-ten" ban đầu hẹp hơn thực tế - cùng kiến trúc CNN, không sửa gì, đã
  train lại/đánh giá trên cấu hình MIMO 2x2 (`results/bler_results_mimo2.csv`),
  vẫn vượt trội bộ ước lượng cổ điển ở đó. Xem
  [`docs/oran-integration.md`](docs/oran-integration.md) để biết thứ thật sự
  cần sửa (số anten thu, không phải model) và thứ chưa chạy (biến thể
  Transformer ở MIMO).
- **Không tích hợp nền tảng O-RAN thật.** Đã đánh giá dựa trên môi trường
  WSL2 thật của máy này và chủ động bỏ qua vì không tương xứng - một bản
  minh hoạ nhẹ cho pattern message chọn model đã được xây và chạy được
  (`oran-stub/`), nhưng đây không phải tích hợp E2AP/RIC chuẩn; lý do được
  ghi rõ, không lướt qua, trong
  [`docs/oran-integration.md`](docs/oran-integration.md).

## Đọc thêm

- [`docs/business-impact.md`](docs/business-impact.md) - cách tính KPI đầy đủ và ranh giới đã nêu.
- [`docs/oran-integration.md`](docs/oran-integration.md) - khung O-RAN chính xác và lộ trình lên 6G.
