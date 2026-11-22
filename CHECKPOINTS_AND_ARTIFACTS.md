# Ghi chú Checkpoint, Feature Cache & Cấu trúc Repo Sau Dọn dẹp

Tài liệu này lưu toàn bộ bản đồ tệp đã sao lưu trên **Google Drive**, cách nạp lại checkpoint hoặc khôi phục/chạy tiếp trên Colab, và danh sách mã nguồn gọn sạch giữ lại trong repository.

---

## 1. Kho lưu trữ sao lưu trên Google Drive (`/content/drive/MyDrive/3DICE_outputs/`)

### 1.1. Bộ đệm đặc trưng đa phương thức (`probe_cache/`, tổng `56 MB`)

| Tệp trên Drive | Dung lượng | Nội dung & Quy cách | Ghi chú sử dụng |
| :--- | :---: | :--- | :--- |
| `probe_cache/drug_chem2d.pt` | `19 MB` | `5,089` thuốc — `ECFP4 + ECFP6 + MACCS + 24 RDKit 2D` (`3263-D`) | Chép vào `/content/cache/probe/` để bỏ qua bước trích xuất RDKit |
| `probe_cache/esm2_650m_pooled.pt` | `18 MB` | `3,286` protein — `ESM-2 650M` (`mean + max`, `2560-D`) | Không cần tải lại checkpoint `2.61 GB` `facebook/esm2_t33_650M_UR50D` |
| `probe_cache/pooled_3d.pt` | `19 MB` | `3,267` protein + `5,089` thuốc — gộp `mean + max` từ `ESM-IF1` & `Uni-Mol` (`1024-D`) | Không cần đọc lại hàng nghìn file `.pt` gốc |
| `probe_cache/probe_signals_summary.json` | `7.0 KB` | Kết quả thăm dò tín hiệu `P1..P5` trước khi thiết kế `Omni3D-DTI` | Bản đối chiếu lịch sử |

### 1.2. Kết quả & Checkpoint `Omni3D-DTI` (`omni3d_results/`)

| Nhóm tệp | Số lượng / Dung lượng | Quy ước đặt tên & Thành phần bên trong |
| :--- | :---: | :--- |
| **Model Checkpoints (`.pt`)** | `24` files | `best_{split}_{arm}_seed0.pt` (trọng số **EMA** tốt nhất theo `Val AUROC` cho `3` bộ chia × `8` biến thể) |
| **Prediction Archives (`.npz`)** | `120` files | `preds_{split}_{arm}_seed{0..4}.npz` (`3` bộ chia × `8` biến thể × `5` seeds); mỗi file chứa `y`, `prob`, `threshold`, `gate_p`, `gate_d` |
| **Tổng hợp số liệu** | `24 KB` / `5.8 KB` / `660 B` | `omni3d_summary.json`, `omni3d_tables.md`, `complexity_summary.json` (đã đồng bộ về repo gốc) |
| **Hình bài báo (`300-DPI`)** | `726 KB` / `185 KB` / `1.1 MB` | `figures/table1_comparison.png`, `figures/table2_complexity.png`, `figures/table3_ablation.png` |

- **`3` bộ chia (`{split}`):** `db_std` (DrugBank chuẩn), `drug_cs` (Drug Cold-start), `prot_cs` (Protein Cold-start).
- **`8` biến thể (`{arm}`):** `Omni3D_DTI`, `Omni3D_wo_3D`, `Omni3D_wo_Seq2D`, `Omni3D_wo_ESM2`, `Omni3D_wo_Chem2D`, `Omni3D_wo_Gate`, `Omni3D_wo_Align`, `Omni3D_wo_ModDrop`.

---

## 2. Lệnh khôi phục, chạy tiếp & nạp lại mô hình

### 2.1. Khôi phục trên Colab để chạy tiếp (Resume tức thì, không train lại từ đầu)

```bash
mkdir -p /content/cache/probe /content/runs/omni3d
cp -r /content/drive/MyDrive/3DICE_outputs/probe_cache/* /content/cache/probe/
cp -r /content/drive/MyDrive/3DICE_outputs/omni3d_results/* /content/runs/omni3d/

# Không truyền --overwrite: tự đọc 120 file .npz đã lưu để tổng hợp lại hoặc bổ sung split mới (như KIBA)
python run_omni3d.py \
  --list-dir 3DICE/lists/db_new \
  --protein-dir /content/embeddings \
  --drug-dir /content/drug_embeddings \
  --cache-dir /content/cache/probe \
  --out-dir /content/runs/omni3d
```

### 2.2. Nạp nhanh một checkpoint `.pt` để suy luận

```python
import torch
from bench.omni.omni_config import OmniRegistry
from bench.omni.omni_model import Omni3dDtiModel

dims = {"p_seq": 2560, "p_3d": 1024, "d_2d": 3263, "d_3d": 1024}
cfg = OmniRegistry.get("Omni3D_DTI")  # hoặc "Omni3D_wo_Gate" cho bản đối xứng 4.89M
model = Omni3dDtiModel(cfg, dims)
state = torch.load("/content/drive/MyDrive/3DICE_outputs/omni3d_results/best_db_std_Omni3D_DTI_seed0.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

### 2.3. Tái tạo bảng biểu & báo cáo Word hoàn toàn ngoại tuyến từ JSON sẵn có

```bash
python render_tables.py --summary omni3d_summary.json --complexity complexity_summary.json --out-dir figures
python build_progress_report.py
```

---

## 3. Cấu trúc Repo Gọn Sạch Hiện Tại

### 3.1. Các tệp & thư mục được giữ lại

- **4 CLI chính ở thư mục gốc:**
  - `run_omni3d.py` — Huấn luyện và đánh giá 5 seeds + 8 biến thể Ablation cho `Omni3D-DTI`.
  - `profile_complexity.py` — Đo `Params`, `Peak VRAM`, `Forward ms`, `Epoch time` trên GPU (`batch_size=32`).
  - `render_tables.py` — Vẽ Table 1, Table 2 (Complexity), Table 3 (Ablation) chuẩn `300-DPI`.
  - `build_progress_report.py` — Sinh `figures/3DICE_vs_Omni3D_DTI.drawio`, `figures/arch_3dice_base.png`, `figures/arch_omni3d_dti.png` và `Bao_cao_Tien_do_3DICE.docx`.
- **Các gói thư viện lõi phục vụ 4 CLI trên:**
  - `bench/omni/` (`18` files) — Kiến trúc `Omni3D-DTI`, huấn luyện, profiler, vẽ bảng/sơ đồ và xuất Word.
  - `bench/data/` (`entity_bundle_cache.py`, `entity_precompute.py`, `entity_record_loader.py`, `pair_dataset.py`) & `bench/models/` (`dice_baselines.py`, `masked_conv_blocks.py`) — Dùng khi đo đối chứng mô hình gốc `3DICE` trong `profile_complexity.py`.
  - `diagnostics/` (`dataset_reader.py`, `embedding_store.py`, `metrics.py`) — Đọc split CSV, embedding `.pt` và tính 7 chỉ số đánh giá Table 1.
  - `research/probe/` (`drug_chem2d_extractor.py`, `esm2_extractor.py`, `pooled_3d_loader.py`, `sequence_drug_collector.py`) — Bộ trích xuất và đọc cache 4 góc nhìn đầu vào.
- **Tài liệu & dữ liệu giữ lại:** `3DICE/` (mã gốc bài báo), `figures/`, `Bao_cao_Tien_do_3DICE.md`, `Bao_cao_Tien_do_3DICE.docx`, `omni3d_summary.json`, `complexity_summary.json`, `improve.md`, `result_reproduce.md`, `btag488.pdf`.

### 3.2. Các mã nguồn thử nghiệm cũ đã xóa bỏ

Đã dọn sạch toàn bộ các script thử nghiệm trung gian không còn nằm trong luồng chính (vẫn truy xuất lại được qua lịch sử Git khi cần): `eval_hypotheses.py`, `extract_ca_coords.py`, `probe_signals.py`, `run_benchmark.py`, `stage_check.py`, `step1_audit.py`, các gói `bench/evaluation/`, `bench/losses/`, `bench/training/`, các bộ phân tích rò rỉ/shortcut lẻ trong `diagnostics/`, mô-đun kiến trúc thử nghiệm cũ trong `bench/models/` & `bench/data/`, và các script `research/` tiền kỳ (`coarse_*`, `cif_*`, `probe_runner.py`, v.v.).
