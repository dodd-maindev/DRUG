# Báo cáo Kết quả Tái hiện Mô hình 3DICE (Google Colab)

- **Bài báo gốc:** *3DICE: interpretable 3D cross-modal learning for drug–target interaction prediction and large-scale drug discovery* (`Bioinformatics`, 2026, 42(7), `btag488`)
- **Repository:** `https://github.com/dodd-maindev/DRUG.git` (thư mục `3DICE/`)
- **Môi trường chạy:** Google Colab (GPU NVIDIA L4, Python 3.13)

---

## 1. Dữ liệu & Tiền xử lý Embeddings (Lưu trên Google Drive)

Do repository GitHub chỉ chứa code và các file chia tập CSV (`lists/db_new/`), toàn bộ cấu trúc 3D và embedding đã được sinh lại trên Colab và lưu trữ lâu dài tại `MyDrive/3DICE_data/`:

| Thành phần | Công cụ / Nguồn | Kết quả | Thời gian chạy | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **Drug Embeddings** | Uni-Mol (`drug/02_embed_drugs.py`) | **5,089 / 5,089** | 6h 19m 00s | Đạt 100% |
| **Protein 3D Structures** | PDB + AlphaFoldDB (`protein/01_fetch_structures_v2.py`) | **3,286 / 3,286** | 14h 18m 46s | Đạt 100% |
| **Protein Embeddings** | ESM-IF1 trên CPU (`protein/02_embed_protein_3D.py`) | **3,267 / 3,285** | 1h 35m 40s | 18 file bị skip do thiếu chain `A` hoặc chứa ligand lạ (`NMM`, `SAH`) |

### Số lượng mẫu thực tế sau khi lọc bỏ 18 protein thiếu embedding

| Kịch bản | Train | Validation | Test |
| :--- | :---: | :---: | :---: |
| **DrugBank (Standard)** | 21,226 / 21,298 | 2,659 / 2,662 | 2,649 / 2,662 |
| **Drug cold-start** | 23,187 | 1,286 | 1,626 |
| **Protein cold-start** | 23,604 | 1,159 | 1,345 |

---

## 2. Cấu hình Huấn luyện (Khớp Bài báo)

- **Hyperparameters:** `EPOCHS = 60`, `BATCH_SIZE = 32`, `LR = 5e-5`, `DROPOUT = 0.3`, `WEIGHT_DECAY = 0`, `Optimizer = Adam`, `Loss = cross_entropy`, `NUM_WORKERS = 2`.
- **Tốc độ huấn luyện trên Colab L4:** ~67–71 giây/epoch (trừ Epoch 1 đọc file `.pt` từ Google Drive lên RAM cache).

---

## 3. Kết quả Tái hiện so với Bài báo (Table 1 — DrugBank & Cold-start)

*(Số liệu bài báo là `mean (std)` qua $n = 5$ lần chạy; số liệu tái hiện là 1 lần chạy trên Colab L4.)*

### 3.1. DrugBank (Standard Split)

| Phiên bản | ACC | PPV (Precision) | TPR (Recall) | F1 | MCC | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bài báo ($n=5$)** | 0.8198 (0.0040) | 0.8086 (0.0118) | 0.8385 (0.0100) | 0.8231 (0.0018) | 0.6403 (0.0074) | 0.8933 (0.0021) | 0.8852 (0.0037) |
| **Colab (Epoch 60)** | 0.8199 | **0.8211** | 0.8193 | 0.8202 | 0.6399 | 0.8915 | **0.8898** |
| **Colab (Best Val — Epoch 57)** | **0.8301** | **0.8181** | **0.8502** | **0.8338** | **0.6607** | **0.8940** | **0.8883** |

- **Nhận xét:** Kết quả tái hiện tại Epoch 60 khớp gần như tuyệt đối với bài báo; khi đánh giá bằng checkpoint tốt nhất trên tập Val (Epoch 57, `val acc = 82.7755%`), mô hình vượt giá trị trung bình của bài báo trên **toàn bộ 7/7 metrics**.

---

### 3.2. Drug Cold-start

| Phiên bản | ACC | PPV (Precision) | TPR (Recall) | F1 | MCC | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bài báo ($n=5$)** | 0.7474 (0.0058) | 0.7471 (0.0207) | 0.7434 (0.0351) | 0.7445 (0.0101) | 0.4913 (0.0130) | 0.8195 (0.0044) | 0.8182 (0.0052) |
| **Colab (Epoch 60 — Best Val)** | 0.7417 | **0.7617** | 0.7036 | 0.7315 | 0.4848 | 0.8118 | 0.8139 |

- **Nhận xét:** Epoch 60 cũng là epoch đạt `val acc` cao nhất (`74.7278%`). Precision (`0.7617`) cao hơn trung bình bài báo, MCC (`0.4848`) và AUPRC (`0.8139`) nằm trong khoảng dao động $\pm 1\text{ std}$ của bài báo.

---

### 3.3. Protein Cold-start

| Phiên bản | ACC | PPV (Precision) | TPR (Recall) | F1 | MCC | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bài báo ($n=5$)** | 0.7476 (0.0103) | 0.7737 (0.0218) | 0.7053 (0.0211) | 0.7375 (0.0053) | 0.5048 (0.0095) | 0.8243 (0.0032) | 0.8297 (0.0035) |
| **Colab (Epoch 60)** | 0.7450 | **0.7978** | 0.6568 | 0.7205 | 0.4979 | 0.8181 | 0.8166 |

- **Nhận xét:** ACC (`0.7450`) và MCC (`0.4979`) nằm trong khoảng $\pm 1\text{ std}$ của bài báo; Precision (`0.7978`) vượt trung bình bài báo. Checkpoint có `val acc` cao nhất là **Epoch 49** (`71.8723%`, đã backup trên Drive, chưa đánh giá riêng trên tập Test).

---

## 4. Hiện trạng Lưu trữ & Các phần Chưa tái hiện

- **Checkpoints & Logs đã backup trên Drive (`MyDrive/3DICE_outputs/`):**
  - DrugBank (Standard): `logs/` + `best_models/` (Epoch 57)
  - Drug cold-start: `drug_cs/` (Epoch 60)
  - Protein cold-start: `protein_cs/` (Epoch 49)
- **Các phần trong bài báo chưa chạy tái hiện:**
  - **Table 1 — KIBA:** Chưa sinh embedding Uni-Mol / ESM-IF1 cho tập KIBA (133 drugs, 2,106 targets).
  - **Table 3 — Ablation Study:** `3DICE w/o CA`, `Prot SA`, `Drug SA`, `ESM-2`.
  - **Table 4 — Clever Hans Test:** `stats/clever_hans/train_clever_hans.py`.
  - **Phân tích giải thích & ổn định:** `faithfulness.py` (Hình 2), `conformer_stability.py` (Mục 3.6).
