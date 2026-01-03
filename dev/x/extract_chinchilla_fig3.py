#!/usr/bin/env python3
"""Extract Figure 3 from Chinchilla paper and detect data points."""

import fitz
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import cv2
from sklearn.cluster import KMeans

OUTPUT_DIR = Path(__file__).parent

# 9 种 Compute 值
COMPUTE_VALUES = [6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21]


def extract_figure3_all():
    pdf_path = "/home/prime/Dropbox/ScalingRelation/2203.15556-Chinchilla.pdf"
    dpi = 1200
    doc = fitz.open(pdf_path)
    page = doc[5]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(25, 25, page.rect.width - 25, 245))
    pix.save(str(OUTPUT_DIR / "fig3_all.png"))
    print("Saved fig3_all.png")
    doc.close()


def detect_circles():
    """用霍夫圆检测找数据点，用颜色聚类确定 Compute 值。"""
    # ROI 范围
    x_min, x_max = 700, 3500
    y_min, y_max = 300, 3100

    # 图例遮罩区域
    legend_x1, legend_x2 = 1174, 1720
    legend_y1, legend_y2 = 1650, 2850

    arr = np.array(Image.open(OUTPUT_DIR / "fig3_all.png"))
    arr_roi = arr[y_min:y_max, x_min:x_max].copy()
    arr_roi_orig = arr[y_min:y_max, x_min:x_max].copy()  # 保留原始颜色
    roi_h, roi_w = arr_roi.shape[:2]

    # 把图例区域变白（避免检测到图例中的圆）
    leg_x1, leg_x2 = legend_x1 - x_min, legend_x2 - x_min
    leg_y1, leg_y2 = legend_y1 - y_min, legend_y2 - y_min
    arr_roi[leg_y1:leg_y2, leg_x1:leg_x2] = 255

    # 转灰度 + 高斯模糊
    gray = cv2.cvtColor(arr_roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 霍夫圆检测
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=16,
        minRadius=15,
        maxRadius=35
    )

    if circles is None:
        print("No circles detected!")
        return

    circles = np.round(circles[0, :]).astype(int)
    print(f"Raw detected: {len(circles)} circles")

    # 过滤：只保留数据区域内的圆（排除坐标轴标签区域）
    # 同时过滤掉颜色太浅的圆（背景误检测）
    valid = []
    for (x, y, r) in circles:
        # 数据区域大约在 y < 2700（排除x轴标签）, x > 350（排除y轴标签）, x < 2650（排除右侧区域）
        if y < 2700 and x > 350 and x < 2650:
            # 检查圆心颜色是否足够深（不是背景）
            x1, x2 = max(0, x-3), min(roi_w, x+4)
            y1, y2 = max(0, y-3), min(roi_h, y+4)
            rgb = arr_roi_orig[y1:y2, x1:x2].mean(axis=(0, 1))
            # 背景是白色/浅灰色，数据点是有色的
            # 排除太浅的颜色（>200），但保留深色点（<60也可以）
            mean_val = rgb.mean()
            if mean_val < 200 and (mean_val < 60 or rgb.max() - rgb.min() > 20):
                valid.append((x, y, r))
    circles = np.array(valid)
    print(f"After filtering: {len(circles)} circles")

    # 转换为 HSV 色彩空间
    arr_hsv = cv2.cvtColor(arr_roi_orig, cv2.COLOR_RGB2HSV)

    colors_rgb = []
    hues = []
    for (x, y, r) in circles:
        # 取圆心附近 7x7 区域的颜色
        x1, x2 = max(0, x-3), min(roi_w, x+4)
        y1, y2 = max(0, y-3), min(roi_h, y+4)
        rgb = arr_roi_orig[y1:y2, x1:x2].mean(axis=(0, 1))
        hsv = arr_hsv[y1:y2, x1:x2].mean(axis=(0, 1))
        colors_rgb.append(rgb)
        hues.append(hsv[0])  # H 分量
    colors_rgb = np.array(colors_rgb)
    hues = np.array(hues).reshape(-1, 1)

    # 用 H 分量聚类
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=20)
    labels = kmeans.fit_predict(hues)

    # 统计每类的数量
    counts = {}
    for i in range(9):
        counts[i] = (labels == i).sum()
    print("Points per cluster:")
    for i, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  Cluster {i}: {c} points")

    # 可视化：在原始图片上画圆检测结果 + 校准线 + 图例框
    img_roi = Image.fromarray(arr_roi_orig)  # 使用原始图片，不变白
    draw = ImageDraw.Draw(img_roi)

    # 获取每个聚类的中心颜色（RGB）
    cluster_centers_rgb = []
    for i in range(9):
        mask = labels == i
        if mask.sum() > 0:
            avg_rgb = colors_rgb[mask].mean(axis=0)
            cluster_centers_rgb.append(tuple(avg_rgb.astype(int)))
        else:
            cluster_centers_rgb.append((128, 128, 128))

    # 画检测到的圆（虚线圆圈）
    for (x, y, r), label in zip(circles, labels):
        rgb = cluster_centers_rgb[label]
        draw_r = r + 5
        for angle in range(0, 360, 20):
            a1 = np.radians(angle)
            a2 = np.radians(angle + 10)
            x1 = int(x + draw_r * np.cos(a1))
            y1 = int(y + draw_r * np.sin(a1))
            x2 = int(x + draw_r * np.cos(a2))
            y2 = int(y + draw_r * np.sin(a2))
            draw.line([(x1, y1), (x2, y2)], fill=rgb, width=3)

    # 画校准参考线（绿色虚线）
    anchor_y_30_local = 1430 - y_min - 2
    anchor_y_20_local = 2882 - y_min + 1
    anchor_x_100m_local = 1576 - x_min
    anchor_x_1b_local = 2206 - x_min
    dash_len, gap_len = 20, 15

    for y_px, label in [(anchor_y_30_local, "3.0"), (anchor_y_20_local, "2.0")]:
        for x in range(0, roi_w, dash_len + gap_len):
            draw.line([(x, y_px), (min(x + dash_len, roi_w), y_px)], fill="green", width=4)
        draw.text((50, y_px + 10), label, fill="green")

    for x_px, label in [(anchor_x_100m_local, "100M"), (anchor_x_1b_local, "1B")]:
        for y in range(0, roi_h, dash_len + gap_len):
            draw.line([(x_px, y), (x_px, min(y + dash_len, roi_h))], fill="green", width=4)
        draw.text((x_px + 10, 100), label, fill="green")

    # 画图例区域框（黑色虚线）
    for x in range(leg_x1, leg_x2, dash_len + gap_len):
        draw.line([(x, leg_y1), (min(x + dash_len, leg_x2), leg_y1)], fill="black", width=3)
        draw.line([(x, leg_y2), (min(x + dash_len, leg_x2), leg_y2)], fill="black", width=3)
    for y in range(leg_y1, leg_y2, dash_len + gap_len):
        draw.line([(leg_x1, y), (leg_x1, min(y + dash_len, leg_y2))], fill="black", width=3)
        draw.line([(leg_x2, y), (leg_x2, min(y + dash_len, leg_y2))], fill="black", width=3)

    img_roi.save(OUTPUT_DIR / "result_chinchilla.png")
    print(f"Saved result_chinchilla.png")

    # 坐标转换：像素 -> 数据值
    # 已知锚点：
    # x: 100M (1e8) @ pixel 1576-700=876, 1B (1e9) @ pixel 2206-700=1506
    # y: 3.0 @ pixel 1430-300=1130, 2.0 @ pixel 2882-300=2582
    anchor_x_100m = 1576 - x_min  # 876
    anchor_x_1b = 2206 - x_min    # 1506
    anchor_y_30 = 1430 - y_min    # 1130
    anchor_y_20 = 2882 - y_min    # 2582

    # 对数坐标 x：log10(P) = log10(1e8) + (x - anchor_x_100m) * (log10(1e9) - log10(1e8)) / (anchor_x_1b - anchor_x_100m)
    # 线性坐标 y：Loss = 3.0 + (y - anchor_y_30) * (2.0 - 3.0) / (anchor_y_20 - anchor_y_30)
    def px_to_params(x_px):
        log_p = np.log10(1e8) + (x_px - anchor_x_100m) * (np.log10(1e9) - np.log10(1e8)) / (anchor_x_1b - anchor_x_100m)
        return 10 ** log_p

    def px_to_loss(y_px):
        return 3.0 + (y_px - anchor_y_30) * (2.0 - 3.0) / (anchor_y_20 - anchor_y_30)

    # 输出数据
    print("\n=== Extracted Data Points ===")
    print(f"{'Cluster':<8} {'P (params)':<15} {'Loss':<8} {'RGB'}")
    data_points = []
    for (x, y, r), label, rgb in zip(circles, labels, colors_rgb):
        p = px_to_params(x)
        loss = px_to_loss(y)
        data_points.append({
            'cluster': label,
            'params': p,
            'loss': loss,
            'rgb': tuple(rgb.astype(int))
        })

    # 按 cluster 分组统计，并按 Hue 值排序（从浅色到深色）
    cluster_info = []
    for i in range(9):
        mask = labels == i
        if mask.sum() > 0:
            avg_hue = hues[mask].mean()
            avg_rgb = colors_rgb[mask].mean(axis=0).astype(int)
            cluster_info.append((i, mask.sum(), avg_hue, tuple(avg_rgb)))

    # 按 hue 排序
    cluster_info.sort(key=lambda x: x[2])

    # 建立 cluster -> compute 映射
    cluster_to_compute = {}
    print("\n=== Summary by Cluster (sorted by Hue) ===")
    print(f"{'Cluster':<8} {'Count':<6} {'Hue':<8} {'RGB':<20} {'Compute'}")
    for idx, (i, count, hue, rgb) in enumerate(cluster_info):
        compute = COMPUTE_VALUES[idx] if idx < len(COMPUTE_VALUES) else 0
        cluster_to_compute[i] = compute
        print(f"Cluster {i}: {count:2d}    H={hue:5.1f}   RGB{rgb}   {compute:.0e}")

    # 导出数据到 CSV
    import csv
    csv_path = OUTPUT_DIR / "extracted_chinchilla_data.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['params', 'loss', 'compute', 'data'])
        for (x, y, r), label in zip(circles, labels):
            p = px_to_params(x)
            loss = px_to_loss(y)
            compute = cluster_to_compute[label]
            data = compute / (6 * p)
            writer.writerow([f"{p:.0f}", f"{loss:.4f}", f"{compute:.0e}", f"{data:.0f}"])
    print(f"\nExported {len(circles)} data points to {csv_path}")

    return circles, labels, data_points


def load_data():
    """读取 CSV 数据。"""
    import csv
    csv_path = OUTPUT_DIR / "extracted_chinchilla_data.csv"
    params_list, loss_list, compute_list, data_list = [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            params_list.append(float(row['params']))
            loss_list.append(float(row['loss']))
            compute_list.append(float(row['compute']))
            data_list.append(float(row['data']))
    return (np.array(params_list), np.array(data_list),
            np.array(loss_list), np.array(compute_list))


def fit_model_1(N, D, L_measured, alpha=0.34, beta=0.28):
    """拟合 L(N,D) = L0 + A/N^alpha + B/D^beta，alpha 和 beta 固定。"""
    from scipy.optimize import minimize

    def objective(params):
        L0, A, B = params
        L_pred = L0 + A / (N ** alpha) + B / (D ** beta)
        log_residual = np.log(L_pred) - np.log(L_measured)
        return np.sum(log_residual ** 2)

    x0 = [1.5, 400, 400]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]  # L0, A, B > 0
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 10000, 'ftol': 1e-12})
    L0, A, B = result.x

    L_pred = L0 + A / (N ** alpha) + B / (D ** beta)
    rmse = np.sqrt(np.mean((L_pred - L_measured) ** 2))

    print(f"\n=== Model 1: L = L0 + A/N^α + B/D^β (α,β fixed) ===")
    print(f"L0 = {L0:.4f}, A = {A:.2f}, B = {B:.2f}")
    print(f"alpha = {alpha} (fixed), beta = {beta} (fixed)")
    print(f"Log loss: {result.fun:.6f}, RMSE: {rmse:.4f}")

    return {'L0': L0, 'A': A, 'B': B, 'alpha': alpha, 'beta': beta, 'rmse': rmse}


def fit_model_2(N, D, L_measured, a0=0.1, b0=0.1):
    """拟合 L(N,D) = A/N^a + B/D^b，a 和 b 也拟合。"""
    from scipy.optimize import minimize

    def objective(params):
        A, B, a, b = params
        L_pred = A / (N ** a) + B / (D ** b)
        log_residual = np.log(L_pred) - np.log(L_measured)
        return np.sum(log_residual ** 2)

    x0 = [10, 10, a0, b0]
    bounds = [(1e-6, None), (1e-6, None), (1e-6, 2.0), (1e-6, 2.0)]  # A, B > 0; 0 < a, b < 2
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 50000, 'ftol': 1e-12})
    A, B, a, b = result.x

    L_pred = A / (N ** a) + B / (D ** b)
    rmse = np.sqrt(np.mean((L_pred - L_measured) ** 2))

    print(f"\n=== Model 2: L = A/N^a + B/D^b (a,b fitted) ===")
    print(f"A = {A:.4f}, B = {B:.4f}")
    print(f"a = {a:.4f}, b = {b:.4f}")
    print(f"Log loss: {result.fun:.6f}, RMSE: {rmse:.4f}")

    return {'A': A, 'B': B, 'a': a, 'b': b, 'rmse': rmse}


def plot_isoloss_contours():
    """用提取的数据拟合 L(N,D) 并画 IsoLoss 等高线图。"""
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    N, D, L_measured, C = load_data()

    print(f"Loaded {len(N)} data points")
    print(f"N range: {N.min():.2e} - {N.max():.2e}")
    print(f"D range: {D.min():.2e} - {D.max():.2e}")
    print(f"L range: {L_measured.min():.4f} - {L_measured.max():.4f}")

    # 拟合两个模型
    result1 = fit_model_1(N, D, L_measured, alpha=0.34, beta=0.28)
    result2 = fit_model_2(N, D, L_measured, a0=0.1, b0=0.1)

    # 用 Model 1 画图
    L0, A, B = result1['L0'], result1['A'], result1['B']
    alpha, beta = result1['alpha'], result1['beta']
    rmse = result1['rmse']

    # 画图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建网格
    C_grid = np.logspace(17, 24, 200)
    N_grid = np.logspace(7, 11, 200)
    C_mesh, N_mesh = np.meshgrid(C_grid, N_grid)
    D_mesh = C_mesh / (6 * N_mesh)

    # 计算 Loss 网格
    L_mesh = L0 + A / (N_mesh ** alpha) + B / (D_mesh ** beta)

    # 画等高线
    levels = np.arange(1.8, 4.0, 0.1)
    contour = ax.contour(C_mesh, N_mesh, L_mesh, levels=levels,
                         cmap='RdYlBu_r', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # 画数据点（按 compute 着色）
    scatter = ax.scatter(C, N, c=np.log10(C), cmap='plasma',
                        s=50, edgecolors='white', linewidths=0.5, zorder=10)

    # Efficient frontier: 对于给定 compute，最优的 N
    # 从 dL/dN = 0 推导：最优 N ∝ C^(beta/(alpha+beta))
    C_frontier = np.logspace(18, 24, 100)
    # 数值求解最优 N
    N_optimal = []
    for c in C_frontier:
        def loss_for_n(log_n):
            n = 10 ** log_n
            d = c / (6 * n)
            return L0 + A / (n ** alpha) + B / (d ** beta)
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(loss_for_n, bounds=(7, 12), method='bounded')
        N_optimal.append(10 ** res.x)
    ax.plot(C_frontier, N_optimal, 'b-', linewidth=2.5, label='Efficient frontier')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training FLOPs (Compute)', fontsize=12)
    ax.set_ylabel('Model Size (Parameters)', fontsize=12)
    ax.set_title('IsoLoss Contours\n' +
                f'$L(N,D) = {L0:.2f} + {A:.0f}/N^{{{alpha}}} + {B:.0f}/D^{{{beta}}}$',
                fontsize=14)
    ax.set_xlim(1e18, 1e24)
    ax.set_ylim(1e8, 1e11)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "isoloss_contours.png", dpi=150)
    print(f"\nSaved isoloss_contours.png")
    plt.close()


if __name__ == "__main__":
    # detect_circles()
    plot_isoloss_contours()
