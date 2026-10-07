# Hướng dẫn chạy Lab 21 trên Google Colab

Notebook đã được cấu hình cho repo cá nhân, tier T4, 2 epochs và full evaluation.

## 1. Trước khi mở Colab

Đẩy các thay đổi mới nhất lên nhánh `main` của GitHub. Colab clone trực tiếp từ:

`https://github.com/ngoclequang1/Day21-Track3-LeQuangNgoc-2A202602664-Finetuning-Lab`

Mở notebook bằng URL:

`https://colab.research.google.com/github/ngoclequang1/Day21-Track3-LeQuangNgoc-2A202602664-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb`

Mỗi lần repo thay đổi, đóng hoặc tải lại tab Colab để notebook mới được nạp.

## 2. Chọn GPU

Trong Colab chọn **Runtime → Change runtime type → T4 GPU**. Chỉ giữ một phiên GPU Colab hoạt động.

Chạy ô Setup và xác nhận đầu ra có:

- GPU là `Tesla T4`.
- VRAM khoảng `14.6 GB`.
- Commit đúng với commit mới nhất trên GitHub.

Nếu GPU là `NONE`, không chạy pipeline; hãy đổi runtime trước.

## 3. Chạy kiểm tra

Chạy ô **Smoke**. Chỉ tiếp tục khi unit tests và kiểm tra dữ liệu đều thành công.

## 4. Chạy pipeline chính

Ở ô **Core pipeline**, giữ nguyên:

```python
COMPUTE_TIER = "T4"
EVAL_LIMIT = ""
STAGES = "nb1 nb2 nb3 nb4 nb5"
```

`EVAL_LIMIT=""` là full evaluation dùng để nộp. Tổng thời gian dự kiến khoảng 100–130 phút trên T4, nhưng có thể lâu hơn khi GPU miễn phí bị chia sẻ.

Nếu Colab ngắt giữa chừng, chạy lại ô Setup rồi đặt `STAGES` từ stage chưa hoàn tất, ví dụ:

```python
STAGES = "nb4 nb5"
```

NB4 tự bỏ qua adapter đã lưu. Không đặt `FORCE_RETRAIN=1` trừ khi muốn huấn luyện lại toàn bộ.

## 5. Kiểm tra trước khi nộp

Chạy ô **Gatekeeper + results**. Yêu cầu cuối cùng là `verify.py` thành công và có đủ:

- `results/mask_proof.json`
- `results/template_check.json`
- `results/token_stats.json`
- `results/baselines_frozen.json`
- `results/runs.csv`
- `results/verdict.json`
- `results/autopsy.json`
- `results/qualitative.json`
- `adapters/correct/adapter_config.json`
- `adapters/correct/adapter_model.safetensors`

Gate `PASSED` hay `FAILED` của fine-tune đều là kết quả hợp lệ; điều kiện là report phải giải thích đúng bằng số đo.

## 6. Tải artefact về máy

Chạy thêm một ô Colab:

```python
import shutil
from google.colab import files

shutil.make_archive(
    "/content/lab21_2A202602664_artifacts",
    "zip",
    root_dir="/content/Day21-Track3-LeQuangNgoc-2A202602664-Finetuning-Lab",
    base_dir="results",
)
files.download("/content/lab21_2A202602664_artifacts.zip")
```

Tải riêng adapter chính nếu cần nộp Option A:

```python
shutil.make_archive(
    "/content/lab21_2A202602664_correct_adapter",
    "zip",
    root_dir="/content/Day21-Track3-LeQuangNgoc-2A202602664-Finetuning-Lab",
    base_dir="adapters/correct",
)
files.download("/content/lab21_2A202602664_correct_adapter.zip")
```

Sau khi tải về, chép `results/` và `adapters/correct/` vào workspace. Dùng các file đó để hoàn thiện mọi số liệu và phân tích trong `submission/REPORT.md`; không dùng số trong `SIMULATION-FINDINGS.md` làm kết quả cá nhân.
