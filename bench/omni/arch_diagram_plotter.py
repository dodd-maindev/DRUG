"""Standalone 300-DPI diagram plotters for original 3DICE and Omni3D-DTI."""

from __future__ import annotations

from bench.omni.diagram_primitives import DiagramPrimitives as dp


class ArchitectureDiagramPlotter:
    """Renders separate Figure 1 (3DICE) and Figure 2 (Omni3D-DTI) PNG diagrams."""

    @classmethod
    def render_base_3dice(cls, out_path: str) -> None:
        """Render standalone Figure 1 showing original 3DICE architecture and bottlenecks."""
        fig, ax = dp.new_canvas(13.8, 3.85, 13.8, 3.85)
        dp.card(ax, 0.18, 0.68, 13.44, 3.02, "#FAFAFA", "#CBD5E1", 1.2)
        ax.text(0.45, 3.42, "Kiến trúc Gốc: 3DICE", fontsize=11.0, fontweight="bold", color="#334155")
        ax.text(2.75, 3.42, "12.87M tham số | 1,851 MB VRAM | 61.36 giây/epoch (NVIDIA L4) | AUROC 0.8933", fontsize=9.0, color="#64748B")

        dp.box(ax, 0.45, 2.12, 2.45, 0.92, "ESM-IF1 (Protein 3D)", "Chỉ dùng khung xương 3D\nBỏ qua danh tính amino acid", "#FEF3C7", "#D97706", 8.2)
        dp.box(ax, 0.45, 0.90, 2.45, 0.92, "Uni-Mol (Thuốc 3D)", "Phụ thuộc cấu dạng 3D\n44.5% nguyên tử H ở cuối", "#FEF3C7", "#D97706", 8.2)
        dp.box(ax, 3.38, 2.12, 2.25, 0.92, "ProteinCNN (1D-CNN)", "Không che chắn padding", "#FEE2E2", "#DC2626", 8.2)
        dp.box(ax, 3.38, 0.90, 2.25, 0.92, "DrugCNN (1D-CNN)", "Rò rỉ padding (lệch 0.737)", "#FEE2E2", "#DC2626", 8.2)
        dp.box(ax, 6.12, 1.42, 2.45, 1.12, "Cross-Attention 2 chiều", "Trao đổi Protein <-> Thuốc\nChưa có cơ chế từ chối cặp âm", "#EDE9FE", "#7C3AED", 8.2)
        dp.box(ax, 9.05, 1.42, 2.30, 1.12, "Gộp trung bình (MeanPool)", "Làm mờ túi gắn kết cục bộ\n1.05M tham số thừa không dùng", "#F1F5F9", "#64748B", 8.1)
        dp.box(ax, 11.82, 1.48, 1.55, 1.00, "Bộ phân loại MLP", "Dự đoán xác suất\ntương tác DTI", "#E2E8F0", "#475569", 8.2)

        dp.arrow(ax, 2.94, 2.58, 3.34, 2.58, "#D97706")
        dp.arrow(ax, 2.94, 1.36, 3.34, 1.36, "#D97706")
        dp.arrow(ax, 5.67, 2.58, 6.08, 2.18, "#DC2626")
        dp.arrow(ax, 5.67, 1.36, 6.08, 1.78, "#DC2626")
        dp.arrow(ax, 8.61, 1.98, 9.01, 1.98, "#7C3AED")
        dp.arrow(ax, 11.39, 1.98, 11.78, 1.98, "#475569")

        dp.legend_bar(ax, 0.14, 13.44, [
            (2.20, "#FEF3C7", "Đầu vào 3D đơn lẻ (Thiếu danh tính / 2D)"),
            (6.15, "#FEE2E2", "Điểm nghẽn / Rò padding"),
            (9.35, "#EDE9FE", "Tương tác chéo & Gộp đặc trưng"),
        ])
        dp.save(fig, out_path)

    @classmethod
    def render_omni3d_dti(cls, out_path: str) -> None:
        """Render standalone Figure 2 showing improved Omni3D-DTI multi-view architecture."""
        fig, ax = dp.new_canvas(13.8, 5.55, 13.8, 5.55)
        dp.card(ax, 0.18, 0.68, 13.44, 4.72, "#F8FBFF", "#93C5FD", 1.4)
        ax.text(0.45, 5.12, "Kiến trúc Cải tiến: Omni3D-DTI", fontsize=11.0, fontweight="bold", color="#0B3B8C")
        ax.text(3.85, 5.12, "5.63M tham số (-2.3x) | 127 MB (-14.5x) | 9.60 giây/epoch (-6.4x) | AUROC 0.9242 (+3.09%)", fontsize=9.0, fontweight="bold", color="#1D4ED8")

        dp.box(ax, 0.42, 4.08, 2.40, 0.66, "ESM-2 650M (Trình tự)", "Danh tính amino acid & tiến hóa", "#DBEAFE", "#0284C7", 8.1)
        dp.box(ax, 0.42, 2.98, 2.40, 0.66, "ESM-IF1 (Hình học 3D)", "Cấu trúc không gian (đủ chuỗi)", "#FEF3C7", "#D97706", 8.1)
        dp.box(ax, 0.42, 1.94, 2.40, 0.66, "Chem2D (Đồ thị & Hóa lý)", "Ổn định, bất biến conformer", "#DBEAFE", "#0284C7", 8.1)
        dp.box(ax, 0.42, 0.84, 2.40, 0.66, "Uni-Mol (Cấu hình 3D)", "Đặc trưng không gian 3D thuốc", "#FEF3C7", "#D97706", 8.1)

        dp.box(ax, 3.26, 4.08, 2.25, 0.66, "Nhánh Protein Trình tự", "Đưa về không gian chung", "#EFF6FF", "#2563EB", 8.1)
        dp.box(ax, 3.26, 2.98, 2.25, 0.66, "Nhánh Protein 3D", "Đưa về không gian chung", "#FFFBEB", "#D97706", 8.1)
        dp.box(ax, 3.26, 1.94, 2.25, 0.66, "Nhánh Thuốc 2D", "Đưa về không gian chung", "#EFF6FF", "#2563EB", 8.1)
        dp.box(ax, 3.26, 0.84, 2.25, 0.66, "Nhánh Thuốc 3D", "Đưa về không gian chung", "#FFFBEB", "#D97706", 8.1)

        dp.box(ax, 6.18, 3.40, 2.20, 0.92, "Hợp nhất Protein", "Cổng cân bằng + Ẩn nhánh\n(Modality Dropout)", "#D1FAE5", "#059669", 8.2)
        dp.box(ax, 6.18, 1.26, 2.20, 0.92, "Hợp nhất Thuốc", "Cổng cân bằng + Ẩn nhánh\n(Modality Dropout)", "#D1FAE5", "#059669", 8.2)
        dp.box(ax, 8.88, 2.13, 2.35, 1.32, "Tương tác Đa góc nhìn", "Kết hợp tương khớp tổng hợp\n+ riêng miền (Seqx2D, 3Dx3D)", "#EDE9FE", "#7C3AED", 8.2)
        dp.box(ax, 11.68, 2.21, 1.72, 1.16, "Dự đoán DTI", "Đầu ra chính + nhánh phụ\nép mọi góc nhìn hữu ích", "#DCFCE7", "#15803D", 8.1)

        for y_c, col in ((4.41, "#0284C7"), (3.31, "#D97706"), (2.27, "#0284C7"), (1.17, "#D97706")):
            dp.arrow(ax, 2.86, y_c, 3.22, y_c, col)
        dp.arrow(ax, 5.55, 4.41, 6.14, 4.02, "#0284C7")
        dp.arrow(ax, 5.55, 3.31, 6.14, 3.68, "#D97706")
        dp.arrow(ax, 5.55, 2.27, 6.14, 1.88, "#0284C7")
        dp.arrow(ax, 5.55, 1.17, 6.14, 1.54, "#D97706")
        dp.arrow(ax, 4.38, 4.03, 4.38, 3.69, "#059669", style="<->", dashed=True)
        dp.arrow(ax, 4.38, 1.89, 4.38, 1.55, "#059669", style="<->", dashed=True)
        ax.text(4.52, 3.86, "Căn chỉnh chéo (InfoNCE)", fontsize=7.6, fontweight="bold", color="#059669", ha="left", va="center")
        ax.text(4.52, 1.72, "Căn chỉnh chéo (InfoNCE)", fontsize=7.6, fontweight="bold", color="#059669", ha="left", va="center")
        dp.arrow(ax, 8.42, 3.86, 8.84, 3.03, "#059669")
        dp.arrow(ax, 8.42, 1.72, 8.84, 2.55, "#059669")
        dp.arrow(ax, 11.27, 2.79, 11.64, 2.79, "#7C3AED")

        dp.legend_bar(ax, 0.14, 13.44, [
            (2.00, "#DBEAFE", "Góc nhìn Trình tự / 2D (Mới)"),
            (4.95, "#FEF3C7", "Góc nhìn Hình học 3D"),
            (7.55, "#D1FAE5", "Căn chỉnh InfoNCE & Cổng"),
            (10.65, "#EDE9FE", "Tương tác Đa góc nhìn"),
        ])
        dp.save(fig, out_path)
