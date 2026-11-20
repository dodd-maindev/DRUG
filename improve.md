# Kế hoạch Cải tiến & Đóng góp Khoa học: **Δ-Interface + CEF** (Nâng cấp từ 3DICE)

- **Nền tảng gốc:** *3DICE: interpretable 3D cross-modal learning for drug–target interaction prediction and large-scale drug discovery* (`Bioinformatics`, 2026, 42(7), `btag488`)
- **Mục tiêu:** Giải quyết tận gốc các nút thắt biểu diễn và giám sát của 3DICE để tạo ra **đóng góp phương pháp luận và bằng chứng khoa học thực sự** (không chỉ lắp ghép kỹ thuật có sẵn).

---

## 1. Bằng chứng Thực nghiệm Stage 0 (Đo trên Google Colab L4)

Toàn bộ kiểm tra trên checkpoint tái hiện 3DICE (`best_models/epoch_57.pt`, DrugBank `AUROC 0.8943`, `AUPRC 0.8927`, `ACC 0.8301`, `MCC 0.6607`) cho thấy 5 sự thật định lượng:

1. **Coarse-Grained Dominance & Gradient Starvation (`shortcut`):**
   - Một mạng MLP 2 lớp cực nhỏ (~200K tham số) chỉ dùng vector mean-pool thô `mean(ESM-IF1) || mean(Uni-Mol)` (`pair_concat`) đã đạt **`AUROC 0.8571` (`MCC 0.5624`)** trên DrugBank — tương đương **91% "skill"** `(AUROC - 0.5)` của toàn bộ mô hình 3DICE `12.9M` tham số!
   - Trên **Drug Cold-start**, chỉ thêm tích phần tử `D ⊙ P` vào vector mean-pool thô (`pair_concat+prod`) đã đạt **`AUROC 0.7641`, `MCC 0.4131`**, vượt cả 3 baseline SOTA của bài báo gốc (`MolTrans 0.3995`, `DLM-DTI 0.4025`, `MocFormer 0.3991`) mà không cần CNN hay Co-Attention.
   - `protein_only(pooled)` đạt **`AUROC 0.6489`** (`db_std`) và **`0.6354`** (`drug_cs`) — trùng khớp kỳ lạ với kết quả `Drug Noise` trong Table 4 Clever Hans của bài báo (`0.6492` và `0.6286`). Ngược lại `degree_only` chỉ đạt `0.50–0.56`.
   - **Ý nghĩa:** Rào cản lớn nhất không phải dung lượng mạng, mà là **gradient starvation** (tín hiệu tương thích họ protein × scaffold dễ học ở mean-pool đã giải thích gần hết nhãn cặp dưới Cross-Entropy biên, triệt tiêu gradient cho nhánh tương tác 3D chi tiết).

2. **Tại sao Softmax Co-Attention trong 3DICE không phân biệt mẫu dương vs mẫu âm:**
   - `attentionp` (`Lp × Ld`) dùng `softmax` theo trục drug nên mọi residue đều có tổng trọng số bằng `1`, không có nút thoát (*null sink*) cho cặp không tương tác, và đầu ra cuối cùng bị `Mean Pooling` đều trên toàn chuỗi. Nhánh drug lại chứa `44.5%` token là Hydro (`H`) nằm ở cuối mảng trước `[SEP]`, nên `Conv1d(k=3)` trượt trên thứ tự 1D bị vô nghĩa về hóa học.

3. **Các lỗi Hygiene đã xác nhận từ code (`tokens` & `leak`):**
   - `Drug` (`300/300` mẫu): `delta = 0`, 100% có `[SEP]` ở hàng cuối (`zero_coord_last = 1.0`), không có `[CLS]` ở đầu → `drug[:-1]` **đúng**.
   - `Protein` (`94/95` mẫu): `L_emb == len(load_coords)` → `ESM-IF1` đã bỏ BOS/EOS sẵn, nên `protein[:-1]` trong `MyDataset` đang **cắt mất 1 amino acid thật cuối chuỗi**.
   - `Mask Leakage`: `DrugCNN` bỏ qua `mask` (`Conv1d` + `bias` + `padding=1`), làm `max |Δmargin| = 0.737` (`mean = 0.030`, `p99 = 0.219`) khi đổi batch size `B=1` sang `B=32`.

4. **Nguồn cấu trúc PDB vs AlphaFold (`strata`) — SẠCH:**
   - `PDB` (`2,351` cặp): `AUROC 0.8937`; `AlphaFold` (`298` cặp): `AUROC 0.8969`; `source-only AUROC = 0.5146` → Không có shortcut từ nguồn cấu trúc.

5. **Khả năng so sánh nội bộ trên cùng Protein (`H-A1` & `L_cond`):**
   - Tỷ lệ protein có đồng thời `>= 1 pos` và `>= 1 neg` là **`98.5%` cặp ở `db_train`** (`2,811/2,893` protein) và **`91.8%`–`95.3%` cặp** trên các tập test (`db_test`, `drug_cs_test`, `prot_cs_test`).
   - Đính chính KIBA: tập `KIBA_pairs_test_stratified.csv` gồm **128–133 proteins** × **~1,663–2,106 drugs** (trong bài báo viết đảo chữ drugs/targets).

---

## 2. Định vị Khoa học & Đóng góp Mới so với SOTA (2023–2026)

| Công trình gần nhất | Trọng tâm của họ | Khoảng trống & Điểm khác biệt của chúng ta |
| :--- | :--- | :--- |
| **3DICE** (*Bioinformatics*, 2026) | Encoder 3D pre-trained + 1D CNN + Co-Attention + Mean Pool | Không dùng tọa độ 3D ở downstream, bị gradient starvation (`CEF ≈ 0.89`), attention không phân biệt pos/neg, leak padding ở CNN. |
| **TAPB** | Causal backdoor adjustment $P(Y \mid D, do(T))$ khử *target prior bias* | Chỉ xử lý bias đơn thực thể (`protein_only ~0.65`), chưa xử lý **coarse pair compatibility (`0.852`)**, không dùng 3D interface geometry. |
| **DebiasedDTA** | Sample reweighting + ensemble khử bias dataset DTA | Không có phân rã thô/mịn, không tạo bản đồ giao diện 3D cục bộ hay ràng buộc phản thực tế. |
| **PCIM-DTA** | Pairwise conditional interaction modeling cho cold-start DTA | Không tách phần dư trên nhánh thô đóng băng + conditional within-protein likelihood + null interface. |
| **ExplainBind / InteractBind** | Đánh giá/dự đoán non-covalent binding residue từ chuỗi | ExplainBind hoạt động không cần 3D; chưa lượng hóa coarse-grained dominance trên mô hình 3D hiện đại như 3DICE. |
| **DrugBAN / PocketDTA / DrugCLIP** | Bilinear attention / cần pocket biết trước | DrugBAN dùng softmax chuẩn (không có null hypothesis & conditional debiasing); PocketDTA/DrugCLIP yêu cầu binding pocket đầu vào. |

### Hai trụ cột đóng góp của bài báo mới
1. **Đóng góp 1 — Benchmark & Chẩn đoán (`CEF` & `Pair-kNN Lookup Audit`):**
   - Đưa ra chỉ số **Coarse-Explained Fraction**: $\text{CEF} = \frac{\text{AUROC}_{\text{pooled}} - 0.5}{\text{AUROC}_{\text{model}} - 0.5}$ và baseline phi tham số **Pair-kNN** để định lượng phần hiệu năng đến từ "tương thích họ / tra cứu lân cận" so với tương tác mịn thực sự.
   - Giới thiệu bộ thước đo đánh giá tương tác mịn: **Within-protein AUROC (`H-A1`)**, **JS divergence giữa bản đồ dương/âm (`H-A2`)**, và **AUPRC định vị túi gắn kết thực nghiệm trên tập giao co-crystal (`H-A3`)**.

2. **Đóng góp 2 — Kiến trúc `Δ-Interface` (Học phần dư giao diện 3D có điều kiện & phản thực tế):**
   - **Phân rã `Δ-learning` ($\ell = b / \tau_b + r$):** Đóng băng nhánh thô $b$ (`C1`: MLP trên $[\bar{e}_p, \bar{e}_d, \bar{e}_p \odot \bar{e}_d]$, dùng 5-fold out-of-fold trên `train` để tránh overfitting của `b` làm triệt tiêu phần dư), buộc toàn bộ nhánh giao diện 3D $r$ chỉ được học phần dư tương tác mịn.
   - **Local 3D Graph Encoder (~0.45M params cho $r$ vs 12.9M của 3DICE):**
     - Protein: 1 lớp kNN graph ($k = 12$) trên tọa độ $C_\alpha$ với $\text{RBF}_{16}(\|x_i - x_j\|)$.
     - Drug: giữ H kèm `is_H` embedding, 1 lớp radius graph ($4.5\text{ Å}$) trên `atomic_coords`.
   - **Low-rank Interface Map kèm tương tác kích thước nhân:**
     $$s_{ia} = \frac{p_i^T q_a}{\sqrt{64}} + c_p(\rho_i)^T c_d(r_g) + \text{mask\_bias}$$
     *(Dùng tích vô hướng $c_p(\rho_i)^T c_d(r_g)$ thay cho phép cộng $\phi_p + \phi_d$ để tránh tự triệt tiêu trong softmax.)*
   - **Giả thuyết Null tường minh ($s_0$):** $r = \text{LogSumExp}_{(i,a)}(s_{ia}) - s_0([\bar{e}_p, \bar{e}_d])$, tách biệt xác suất *tồn tại giao diện* $\sigma(r)$ khỏi phân phối vị trí có điều kiện $\pi_{ia} = \text{softmax}_{(i,a)}(s_{ia})$.
   - **Conditional Within-Entity Likelihood ($L_{\text{cond}}$) dùng Entity-Grouped BatchSampler:** Loại bỏ thiên lệch họ/mức nền từng protein và từng thuốc ra khỏi gradient của $r$.
   - **Ràng buộc Phản thực tế & Độ tụ Không gian ($L_{\text{nec}} + L_{\text{spec}} + L_{\text{compact}}$):**
     - $L_{\text{nec}}$ & $L_{\text{spec}}$ (chỉ trên $y=1$, **chạy lại cả 1 lớp kNN encoder** sau khi che top-$m$ hoặc random-$m$ ngoài top-$m$ để chống rò rỉ thông tin qua lân cận đồ thị).
     - $L_{\text{compact}}$ (**chỉ tính trên $y=1$**) với bán kính thích ứng theo ligand $R_d = r_g(d) + 6.0\text{ Å}$ và phạt $\text{ReLU}(\sqrt{\text{Var}_\pi(x) + \varepsilon} - R_d)^2$.

---

## 3. Các Giả thuyết Kiểm chứng Được (Falsifiable Hypotheses)

| Mã | Giả thuyết | Phép đo | Tiêu chuẩn bác bỏ |
| :--- | :--- | :--- | :--- |
| **H-A1** | Cải thiện xếp hạng trong cùng protein | Mean within-protein AUROC (protein có $\ge 1$ pos & $\ge 1$ neg), paired bootstrap | Không vượt `C2` (Lego++) $\ge 0.02$ |
| **H-A2** | Null + $L_{\text{cond}}$ làm attention dương/âm tách biệt | Jensen–Shannon divergence giữa $\pi(\cdot \mid d^+)$ và $\pi(\cdot \mid d^-)$ cùng protein | Không lớn hơn `A2` rõ rệt |
| **H-A3** | Định vị pocket đúng mà không cần nhãn pocket lúc train | AUPRC residue trên tập con cặp dương DrugBank trùng ligand đồng kết tinh trong `.cif` | Không vượt attention của `A2` |
| **H-A4** | Đóng góp thật của $r$ ngoài hygiene/recipe | $\Delta\text{AUROC}$ của `M1` so với `A2` và `C1` | Nếu tăng AUROC tổng nhưng H-A1 không tăng → lợi ích đến từ recipe |
| **H-A5** | Cold-start vượt trội phần tương thích thô | MCC Drug cold-start so với `C1` (`pair_concat+prod`) | Không tăng $\ge 0.02$ |

---

## 4. Ma trận Đối chứng Thực nghiệm (`A0 → M1`)

- **`A0`:** 3DICE gốc (tái hiện 5 seed, `trim_protein=True`, unmasked CNN, `Adam lr=5e-5`, chọn theo `val_acc`).
- **`A1`:** `A0` + vá hygiene Stage 1 (bỏ `protein[:-1]`, mask-aware `DrugCNN`, xóa `1.05M` dead params).
- **`A2`:** `A1` + training recipe (`AdamW lr=3e-4`, cosine, grad clip `1.0`, EMA `0.999`, label smoothing `0.05`, chọn theo `val_auroc`).
- **`C1`:** Nhánh thô $b$ đơn lẻ (`MLP([ē_p, ē_d, ē_p ⊙ ē_d])` cùng recipe) → tính **CEF**.
- **`C2` ("Lego++"):** Kiến trúc ghép chuẩn (cùng `ProteinKnnEncoder` + `DrugRadiusEncoder` + Pair Softmax MIL Pooling, **không** dùng `Δ-learning`, `L_cond`, `s_0`, `L_nec/spec/compact`) → kiểm soát dung lượng để chứng minh đóng góp thuần $\Delta = \text{M1} - \text{C2}$.
- **`M1` & `M1−x`:** `Δ-Interface` đầy đủ + ablation bỏ từng thành phần (`M1_no_s0`, `M1_no_cond`, `M1_no_cf`, `M1_no_compact`, `M1_no_stopgrad`, `M1_no_size`).

---

## 5. Lộ trình Triển khai & Kết quả Đã Chốt

### Pha 1 — Audit Hướng B + Chuẩn bị Tọa độ 3D *(Đã hoàn thành 100% trên Colab L4)*
1. **[step1_audit.py](file:///c:/Workspace_2/Drug/step1_audit.py)** (gọi gói [research/](file:///c:/Workspace_2/Drug/research/__init__.py)):
   - `pair-knn (k=10)`: `AUROC 0.7054 / MCC 0.3110` (`db_std`), `0.6967 / 0.2374` (`drug_cs`), `0.6458 / 0.2128` (`prot_cs`).
   - `c1-coarse` (`5` seed): `db_std` **`AUROC 0.8516 ± 0.0024, MCC 0.5467` (`CEF = 0.892`)**; `drug_cs` **`AUROC 0.7636 ± 0.0066, MCC 0.4044` (`CEF = 0.845`)**; `prot_cs` **`AUROC 0.7709 ± 0.0030, MCC 0.3974` (`CEF = 0.852`)**. Vượt `Pair-kNN` +0.07–0.15 AUROC → `C1` học tương thích liên miền chứ không chỉ tra cứu lân cận.
   - `pdb-overlap`: Quét `3,286` cấu trúc (`2,758` PDB, `1,976` có non-polymer ligand, `1,316/1,318` CCD có SMILES hợp lệ) → **`835` cặp dương khớp chính xác InChIKey-14** và **`1,060` cặp Tanimoto ≥ 0.70** (`pdb_overlap_pairs.csv`) làm ground truth cho `H-A3`.
2. **[extract_ca_coords.py](file:///c:/Workspace_2/Drug/extract_ca_coords.py)**:
   - Trích xuất khớp **`3,267 / 3,267`** protein (`mismatched = 0, failed = 0`), lưu `(L_p, 3)` `ca_coords`, `valid_mask`, và `pLDDT` ra `ca_coords/{uid}.npz`.

### Pha 2 & Pha 3 — Khung Benchmark & Kiểm định `M1` *(Đã triển khai đầy đủ, sẵn sàng chạy Colab)*
- Huấn luyện đối chứng tại [run_benchmark.py](file:///c:/Workspace_2/Drug/run_benchmark.py) (gói [bench/](file:///c:/Workspace_2/Drug/bench/__init__.py)): chạy `A0`, `A1`, `A2`, `C2`, `M1`, và 6 ablation `M1_*` trên Track A (split gốc) & Track B (degree-matched negatives).
- Kiểm chứng khoa học tại [eval_hypotheses.py](file:///c:/Workspace_2/Drug/eval_hypotheses.py): tự động xuất bảng tổng hợp `H-A1..H-A5` (Within-protein AUROC + bootstrap 95% CI, Pos/Neg JS divergence, Padding Leak check, và Zero-shot Pocket Localization AUPRC trên `835` phức hợp co-crystal thật).
