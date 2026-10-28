import numpy as np, os
OUT = os.path.dirname(os.path.abspath(__file__))
INK = "#222222"; BLUE = "#1f77b4"; RED = "#d62728"; GREEN = "#2ca02c"; GREY = "#9a9a9a"; ORANGE = "#ff7f0e"


def svg(w, h, body, name):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="Arial, sans-serif">\n<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')
    open(os.path.join(OUT, name), "w").write(s)
    print(name, len(s))


def text(x, y, t, size=14, fill=INK, anchor="middle", weight="normal"):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{t}</text>'


def dots(pts, color, r=3):
    d = "".join(f"M{x:.0f} {y:.0f}h0" for x, y in pts)
    return f'<path d="{d}" stroke="{color}" stroke-width="{2*r}" stroke-linecap="round" fill="none"/>'


# ---------------------------------------------------------- 1. shape features
rng = np.random.default_rng(2)
n = 150
pole = np.c_[rng.normal(0, .05, n), rng.normal(0, .05, n), rng.uniform(-1, 1, n)]
road = np.c_[rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.normal(0, .02, n)]
bush = rng.normal(0, .35, (n, 3))
sets = [("Pole (linear)", pole, RED), ("Road (planar)", road, BLUE), ("Bush (scattered)", bush, GREEN)]
body = [text(450, 28, "Eigenvalues of a point's neighbourhood describe its shape", 18, weight="bold")]
feats = []
for k, (name, P, col) in enumerate(sets):
    X = P - P.mean(0)
    lam = np.sort(np.linalg.eigvalsh(X.T @ X / (len(P) - 1)))[::-1]
    s = np.sqrt(lam)
    a1, a2, a3 = (s[0] - s[1]) / s[0], (s[1] - s[2]) / s[0], s[2] / s[0]
    feats.append((name, lam, (a1, a2, a3)))
    ox = 150 + k * 300
    # isometric projection
    pr = np.c_[(P[:, 0] - P[:, 1]) * 0.866, (P[:, 0] + P[:, 1]) * 0.5 - P[:, 2]]
    body.append(dots(np.array([ox, 145]) + 48 * pr, col, 2.2))
    body.append(text(ox, 62, name, 15, col, weight="bold"))
    # bars of normalised lambdas
    by = 300
    for i in range(3):
        hgt = 80 * lam[i] / lam[0]
        bx = ox - 75 + i * 55
        body.append(f'<rect x="{bx}" y="{by - hgt:.0f}" width="36" height="{max(hgt,1):.0f}" fill="{col}"/>')
        body.append(text(bx + 18, by + 18, f"λ{i+1}", 13))
    body.append(f'<line x1="{ox-85}" y1="{by}" x2="{ox+95}" y2="{by}" stroke="{INK}"/>')
    body.append(text(ox, by + 44, f"linearity {a1:.2f}", 13, RED if a1 == max(a1, a2, a3) else INK))
    body.append(text(ox, by + 62, f"planarity {a2:.2f}", 13, BLUE if a2 == max(a1, a2, a3) else INK))
    body.append(text(ox, by + 80, f"scattering {a3:.2f}", 13, GREEN if a3 == max(a1, a2, a3) else INK))
body.append(text(450, 410, "σᵢ = √λᵢ.  linearity = (σ1−σ2)/σ1,  planarity = (σ2−σ3)/σ1,  scattering = σ3/σ1  (they sum to 1)", 13, GREY))
svg(900, 430, "\n".join(body), "shape-features.svg")
for f in feats:
    print(f[0], np.round(f[1] / f[1][0], 3), np.round(f[2], 2))

# ---------------------------------------------------------- 2. IoU
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i, u, i / u


pairs = [([0, 0, 4, 4], [2, 0, 6, 4]), ([0, 0, 4, 4], [1, 1, 5, 5]), ([0, 0, 4, 4], [0, 0, 4, 3])]
body = [text(450, 28, "Intersection over Union: how much two boxes overlap", 18, weight="bold")]
sc = 28
for k, (A, B) in enumerate(pairs):
    i, u, v = iou(A, B)
    ox, oy = 60 + k * 290, 70
    for bx, col in [(A, RED), (B, BLUE)]:
        body.append(f'<rect x="{ox + bx[0]*sc}" y="{oy + bx[1]*sc}" width="{(bx[2]-bx[0])*sc}" height="{(bx[3]-bx[1])*sc}" '
                    f'fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="2.5"/>')
    x1 = max(A[0], B[0]); y1 = max(A[1], B[1]); x2 = min(A[2], B[2]); y2 = min(A[3], B[3])
    body.append(f'<rect x="{ox + x1*sc}" y="{oy + y1*sc}" width="{(x2-x1)*sc}" height="{(y2-y1)*sc}" fill="{"#9467bd"}" fill-opacity="0.45"/>')
    verdict = "kept (IoU ≤ 0.5)" if v <= 0.5 else "suppressed (IoU > 0.5)"
    body.append(text(ox + 85, 245, f"overlap {i} / union {u}", 14))
    body.append(text(ox + 85, 266, f"IoU = {v:.2f}", 16, weight="bold"))
    body.append(text(ox + 85, 288, f"NMS at 0.5: {verdict}", 13, GREEN if v <= 0.5 else RED))
body.append(text(450, 320, "Both boxes are 4×4 units; the purple area is the intersection.", 13, GREY))
svg(900, 335, "\n".join(body), "iou-examples.svg")

# ---------------------------------------------------------- 3. oriented boxes
rng = np.random.default_rng(7)
L, W = 4.6, 1.9
th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def outline(sides, m=70):
    pts = []
    t = rng.uniform(0, 1, m)
    for sd in sides:
        if sd == 0: pts += [[-L/2 + L*u, -W/2] for u in t]
        if sd == 1: pts += [[L/2, -W/2 + W*u] for u in t[:m//2]]
        if sd == 2: pts += [[-L/2 + L*u, W/2] for u in t]
        if sd == 3: pts += [[-L/2, -W/2 + W*u] for u in t[:m//2]]
    P = np.array(pts) + rng.normal(0, .04, (len(pts), 2))
    return P @ R.T


def boxes(P):
    lo, hi = P.min(0), P.max(0)
    c = P.mean(0); X = P - c
    w, V = np.linalg.eigh(X.T @ X)
    e1, e2 = V[:, 1], V[:, 0]
    p1, p2 = X @ e1, X @ e2
    corners = [c + a * e1 + b * e2 for a, b in [(p1.min(), p2.min()), (p1.max(), p2.min()), (p1.max(), p2.max()), (p1.min(), p2.max())]]
    head = np.degrees(np.arctan2(e1[1], e1[0])) % 180
    return (lo, hi), np.array(corners), head, p1.max() - p1.min(), p2.max() - p2.min()


body = [text(450, 28, "PCA gives an oriented box, but only when it sees the whole object", 18, weight="bold")]
for k, (sides, name) in enumerate([((0, 1, 2, 3), "All four sides seen"), ((0, 3), "Only two sides seen (L-shape)")]):
    P = outline(sides)
    (lo, hi), cor, head, ln, wd = boxes(P)
    ox, oy, s = 225 + k * 450, 200, 52
    tr = lambda q: (ox + s * q[0], oy - s * q[1])
    body.append(text(ox, 62, name, 15, weight="bold"))
    a, b = tr(lo), tr(hi)
    body.append(f'<rect x="{a[0]:.0f}" y="{b[1]:.0f}" width="{b[0]-a[0]:.0f}" height="{a[1]-b[1]:.0f}" fill="none" stroke="{GREY}" stroke-width="2" stroke-dasharray="6 4"/>')
    poly = " ".join(f"{tr(q)[0]:.0f},{tr(q)[1]:.0f}" for q in cor)
    body.append(f'<polygon points="{poly}" fill="none" stroke="{RED}" stroke-width="2.5"/>')
    body.append(dots([tr(q) for q in P], INK, 2))
    body.append(text(ox, 330, f"PCA heading {head:.0f}° (true 30°)", 14, RED if k else INK))
    body.append(text(ox, 350, f"PCA box {ln:.1f} × {wd:.1f} m (true {L} × {W} m)", 14))
body.append(f'<line x1="300" y1="380" x2="330" y2="380" stroke="{GREY}" stroke-width="2" stroke-dasharray="6 4"/>')
body.append(text(338, 385, "axis-aligned box", 13, anchor="start"))
body.append(f'<line x1="480" y1="380" x2="510" y2="380" stroke="{RED}" stroke-width="2.5"/>')
body.append(text(518, 385, "PCA-oriented box", 13, anchor="start"))
svg(900, 400, "\n".join(body), "oriented-boxes.svg")
