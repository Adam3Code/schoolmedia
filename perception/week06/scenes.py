"""Manim scenes for Perception for Autonomous Systems, Week 6 (Clustering & Regression).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
All scenes compute the real algorithm with NumPy, so the motion shows what the method actually does.
"""
from manim import *
import numpy as np

Text.set_default(font="Inter")

BLUE_C1, RED_C1, GREEN_C1, PURPLE_C1, ORANGE_C1 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"
PAL = [BLUE_C1, RED_C1, GREEN_C1, PURPLE_C1, ORANGE_C1]
GREY_C1 = "#9a9a9a"
INK = "#222222"


def title(text):
    return Text(text, color=INK, font_size=34, weight=BOLD).to_edge(UP, buff=0.35)


def caption(text):
    return Text(text, color=INK, font_size=24).to_edge(DOWN, buff=0.35)


class BaseScene(Scene):
    def setup(self):
        self.camera.background_color = WHITE
        self.cap = caption(" ")
        self.add(self.cap)

    def say(self, text, run_time=0.6):
        new = caption(text)
        self.play(FadeOut(self.cap, shift=UP * 0.15), FadeIn(new, shift=UP * 0.15), run_time=run_time)
        self.cap = new


# ---------------------------------------------------------------- 1. Least squares
class LeastSquares(BaseScene):
    def construct(self):
        rng = np.random.default_rng(4)
        self.add(title("Least squares: make the squared errors small"))
        xs = np.linspace(-4.5, 4.5, 9)
        ys = 0.45 * xs - 0.2 + rng.normal(0, 0.55, xs.size)
        X = np.c_[xs, np.ones_like(xs)]
        a_opt, b_opt = np.linalg.lstsq(X, ys, rcond=None)[0]

        a = ValueTracker(0.0)
        b = ValueTracker(0.9)
        dots = VGroup(*[Dot([x, y, 0], radius=0.08, color=BLUE_C1) for x, y in zip(xs, ys)])

        def line():
            return Line([-5.5, a.get_value() * -5.5 + b.get_value(), 0],
                        [5.5, a.get_value() * 5.5 + b.get_value(), 0], color=RED_C1, stroke_width=4)

        def squares():
            g = VGroup()
            for x, y in zip(xs, ys):
                yh = a.get_value() * x + b.get_value()
                r = y - yh
                side = abs(r)
                if side < 1e-3:
                    continue
                sq = Square(side_length=side, stroke_color=RED_C1, stroke_width=1.5,
                            fill_color=RED_C1, fill_opacity=0.18)
                sq.move_to([x + side / 2, (y + yh) / 2, 0])
                g.add(sq)
            return g

        def mse():
            r = ys - (a.get_value() * xs + b.get_value())
            return float(np.mean(r ** 2))

        ln = always_redraw(line)
        sq = always_redraw(squares)
        label = Text("MSE =", color=INK, font_size=28)
        val = always_redraw(lambda: DecimalNumber(mse(), num_decimal_places=3, color=INK, font_size=30)
                            .next_to(label, RIGHT, buff=0.15))
        readout = VGroup(label).to_corner(UL, buff=0.5).shift(DOWN * 0.7)

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08), run_time=1.2)
        self.say("Data points (x, t). Try a line y = ax + c.")
        self.play(Create(ln), FadeIn(readout), FadeIn(val))
        self.add(sq)
        self.say("Each red square has area = (t − y)², the squared error of one point.")
        self.wait(1.2)
        self.say("Least squares turns a and c until the total area is as small as possible.")
        self.play(a.animate.set_value(0.2), b.animate.set_value(-0.7), run_time=1.6)
        self.play(a.animate.set_value(a_opt), b.animate.set_value(b_opt), run_time=2.2, rate_func=smooth)
        sol = MathTex(r"\hat b = (X^\top X)^{-1} X^\top t", color=INK, font_size=40)
        sol.next_to(readout, DOWN, buff=0.45).align_to(readout, LEFT)
        box = SurroundingRectangle(sol, color=GREEN_C1, buff=0.15)
        self.say("The minimum has a closed form: the normal equations.")
        self.play(Write(sol), Create(box))
        self.wait(1.5)

        # Part 2: near-vertical data, vertical vs perpendicular residuals
        self.remove(sq)
        self.play(*[FadeOut(m) for m in [dots, ln, readout, val, sol, box]])
        t = np.linspace(-1, 1, 11)
        px = 0.35 * t + rng.normal(0, 0.28, t.size)
        py = 2.6 * t
        pts = np.c_[px, py]
        dots2 = VGroup(*[Dot([x, y, 0], radius=0.08, color=BLUE_C1) for x, y in pts])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots2], lag_ratio=0.06), run_time=1.0)
        self.say("A near-vertical surface, like a wall seen from the side.")
        A, B = np.polyfit(px, py, 1)
        ys_l = np.array([-3.0, 3.0])
        ols = Line([(ys_l[0] - B) / A, ys_l[0], 0], [(ys_l[1] - B) / A, ys_l[1], 0], color=RED_C1, stroke_width=4)
        res_v = VGroup(*[DashedLine([x, y, 0], [x, A * x + B, 0], color=RED_C1, stroke_width=2) for x, y in pts])
        self.play(Create(ols), Create(res_v))
        self.say("Ordinary least squares measures errors vertically, so the line tilts wrongly.")
        self.wait(1.4)
        c = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - c)
        d = Vt[0]
        tls = Line([*(c - 3.2 * d), 0], [*(c + 3.2 * d), 0], color=GREEN_C1, stroke_width=4)
        res_p = VGroup(*[DashedLine([x, y, 0], [*(c + ((np.array([x, y]) - c) @ d) * d), 0],
                                    color=GREEN_C1, stroke_width=2) for x, y in pts])
        self.say("Total least squares (PCA) measures them perpendicular to the line.")
        self.play(ReplacementTransform(ols, tls), ReplacementTransform(res_v, res_p), run_time=1.6)
        note = Text("line direction: largest-eigenvalue eigenvector\nnormal: smallest-eigenvalue eigenvector", color=GREEN_C1, font_size=20)
        note.to_edge(RIGHT, buff=0.4).shift(UP * 0.8)
        self.play(FadeIn(note))
        self.say("Same computation as normal estimation in Week 5: works in any orientation.")
        self.wait(2)


# ---------------------------------------------------------------- 2. k-means
class KMeansIterations(BaseScene):
    def construct(self):
        rng = np.random.default_rng(1)
        self.add(title("k-means: assign, then move the centres"))
        true_c = [(-3.6, 0.5), (3.2, 0.7), (0.2, -1.6)]
        pts = np.vstack([rng.normal(c, 0.65, (28, 2)) for c in true_c])
        dots = VGroup(*[Dot([x, y, 0], radius=0.07, color=GREY_C1) for x, y in pts])
        eq = MathTex(r"J=\sum_k \sum_{n\in k}\lVert x_n-\mu_k\rVert^2", color=INK, font_size=34)
        eq.move_to([4.6, -2.0, 0])
        jlab = Text("J =", color=INK, font_size=26).next_to(eq, DOWN, buff=0.25).align_to(eq, LEFT)
        jt = ValueTracker(0)
        jnum = always_redraw(lambda: DecimalNumber(jt.get_value(), num_decimal_places=0, color=INK, font_size=28)
                             .next_to(jlab, RIGHT, buff=0.15))
        self.play(FadeIn(dots), Write(eq))

        def J(l, m):
            return float(((pts - m[l]) ** 2).sum())

        def make_cross(m, col):
            return VGroup(Cross(scale_factor=0.22, stroke_color=BLACK, stroke_width=10),
                          Cross(scale_factor=0.22, stroke_color=col, stroke_width=6)).move_to([*m, 0])

        def run(init, intro, fast=False):
            mu = np.array(init, float)
            crosses = [make_cross(m, PAL[k]) for k, m in enumerate(mu)]
            self.say(intro)
            self.play(*[FadeIn(c, scale=1.5) for c in crosses], *[d.animate.set_color(GREY_C1) for d in dots])
            lab = None
            rt = 0.6 if fast else 0.9
            for it in range(1, 15):
                new = np.argmin(((pts[:, None] - mu[None]) ** 2).sum(-1), 1)
                if lab is not None and np.array_equal(new, lab):
                    break
                lab = new
                if not fast or it == 1:
                    self.say(f"Iteration {it}, step 1: give each point to its nearest centre.")
                self.play(*[d.animate.set_color(PAL[l]) for d, l in zip(dots, lab)],
                          jt.animate.set_value(J(lab, mu)), run_time=rt)
                mu = np.array([pts[lab == k].mean(0) for k in range(3)])
                if not fast or it == 1:
                    self.say(f"Iteration {it}, step 2: move each centre to the mean of its points.")
                self.play(*[c.animate.move_to([*m, 0]) for c, m in zip(crosses, mu)],
                          jt.animate.set_value(J(lab, mu)), run_time=rt)
            return crosses, J(lab, mu)

        self.add(jlab, jnum)
        crosses, j1 = run([[-5.5, 2.4], [-4.2, 2.6], [-2.8, 2.4]], "Start: k = 3 centres, all placed near the left.")
        self.say(f"Converged, but badly: two centres share one blob. J = {j1:.0f} is only a local minimum.")
        self.play(*[Indicate(c, color=c[1].get_color(), scale_factor=1.3) for c in crosses])
        self.wait(1.2)
        self.play(*[FadeOut(c) for c in crosses])
        crosses, j2 = run([[-4.8, -2.5], [-3.2, 2.6], [-1.6, 2.6]], "Restart from a different random start.", fast=True)
        self.say(f"Converged with J = {j2:.0f}, much lower. Keep the run with the lowest J.")
        self.wait(1.2)
        self.say("This is why k-means++ initialisation and several restarts are used in practice.")
        self.wait(2)


# ---------------------------------------------------------------- 3. Mean shift
class MeanShiftModes(BaseScene):
    def construct(self):
        rng = np.random.default_rng(7)
        self.add(title("Mean shift: climb to the densest spot"))
        blobs = [((-3.6, 0.2), 0.7, 45), ((3.2, 0.4), 0.8, 50), ((0.0, -1.5), 0.55, 30)]
        pts = np.vstack([rng.normal(c, s, (n, 2)) for c, s, n in blobs])
        h = 1.4
        dots = VGroup(*[Dot([x, y, 0], radius=0.06, color=GREY_C1) for x, y in pts])
        self.play(FadeIn(dots))
        eq = MathTex(r"m(x)=\frac{\sum_i K(x-x_i)\,x_i}{\sum_i K(x-x_i)}-x", color=INK, font_size=32)
        eq.move_to([0, 2.45, 0])
        self.play(Write(eq))

        def shift(x):
            inside = np.linalg.norm(pts - x, axis=1) <= h
            return pts[inside].mean(0), inside

        # one window, step by step
        x = pts[:45][np.argmax(pts[:45, 0] - 0.5 * pts[:45, 1])].copy()  # an edge point of the left blob
        win = Circle(radius=h, color=PURPLE_C1, stroke_width=3).move_to([*x, 0])
        centre = Dot([*x, 0], radius=0.09, color=PURPLE_C1)
        self.say("Put a window (radius h) somewhere. Only points inside it count.")
        self.play(Create(win), FadeIn(centre))
        trail = [x.copy()]
        for step in range(10):
            m, inside = shift(x)
            if np.linalg.norm(m - x) < 0.06:
                break
            hi = VGroup(*[dots[i] for i in np.where(inside)[0]])
            arrow = Arrow([*x, 0], [*m, 0], buff=0, color=ORANGE_C1, stroke_width=6,
                          max_tip_length_to_length_ratio=0.35)
            if step == 0:
                self.say("Mean of the points inside is pulled towards the denser side.")
                self.play(hi.animate.set_color(INK), run_time=0.5)
                self.play(GrowArrow(arrow))
                self.say("Mean shift vector: move the window to that mean. Repeat.")
                self.wait(0.6)
            else:
                self.play(hi.animate.set_color(INK), GrowArrow(arrow), run_time=0.35)
            seg = Line([*x, 0], [*m, 0], color=PURPLE_C1, stroke_width=3)
            self.play(win.animate.move_to([*m, 0]), centre.animate.move_to([*m, 0]), Create(seg),
                      FadeOut(arrow), hi.animate.set_color(GREY_C1), run_time=0.5 if step else 0.9)
            x = m
            trail.append(x.copy())
        self.say("Steps shrink near the peak: the window stops at a mode of the density.")
        self.wait(1.0)
        self.play(FadeOut(win))

        # every point climbs
        def climb(x0):
            path = [x0]
            x = x0.copy()
            for _ in range(60):
                m, _ = shift(x)
                path.append(m)
                if np.linalg.norm(m - x) < 1e-4:
                    break
                x = m
            return np.array(path)

        paths = [climb(p.copy()) for p in pts]
        ends = np.array([p[-1] for p in paths])
        modes = []
        lab = np.zeros(len(pts), int)
        for i, e in enumerate(ends):
            for k, md in enumerate(modes):
                if np.linalg.norm(e - md) < h / 2:
                    lab[i] = k
                    break
            else:
                modes.append(e)
                lab[i] = len(modes) - 1
        self.say("Now start a window at every point and let them all climb.")
        moving = [i for i, p in enumerate(paths) if np.linalg.norm(p[-1] - p[0]) > 0.05]
        movers = VGroup(*[Dot([*paths[i][0], 0], radius=0.05, color=PAL[lab[i]]) for i in moving])
        trails = VGroup(*[VMobject(stroke_color=PAL[lab[i]], stroke_width=1.2, stroke_opacity=0.6)
                          .set_points_as_corners([[*q, 0] for q in paths[i]]) for i in moving])
        self.add(movers)
        self.play(*[MoveAlongPath(mv, tr) for mv, tr in zip(movers, trails)],
                  *[Create(tr) for tr in trails], run_time=2.6, rate_func=smooth)
        stars = VGroup(*[Star(5, outer_radius=0.22, color=BLACK, fill_color=PAL[k], fill_opacity=1, stroke_width=2)
                         .move_to([*md, 0]) for k, md in enumerate(modes)])
        self.play(FadeIn(stars, scale=1.6), FadeOut(movers))
        self.say(f"{len(modes)} modes found, without choosing the number in advance.")
        self.play(*[d.animate.set_color(PAL[l]) for d, l in zip(dots, lab)], FadeOut(trails), run_time=1.2)
        self.say("Cluster = all points whose climb ends at the same mode (its attraction basin).")
        self.wait(2)


# ---------------------------------------------------------------- 4. DBSCAN
class DBSCANGrowth(BaseScene):
    def construct(self):
        rng = np.random.default_rng(3)
        self.add(title("DBSCAN: grow clusters through dense regions"))
        n = 70
        t = rng.uniform(0, np.pi, n)
        A = np.c_[np.cos(t), np.sin(t)]
        t = rng.uniform(0, np.pi, n)
        B = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
        P = np.vstack([A, B]) + rng.normal(0, 0.07, (2 * n, 2))
        noise = rng.uniform([-1.2, -0.8], [2.2, 1.2], (10, 2))
        P = np.vstack([P, noise])
        S = 2.6
        P = (P - [0.5, 0.25]) * S + [0, -0.35]
        eps, minpts = 0.24 * S, 5
        D = np.linalg.norm(P[:, None] - P[None], axis=2)
        nb = [np.where(D[i] <= eps)[0] for i in range(len(P))]
        core = np.array([len(x) >= minpts for x in nb])
        dots = VGroup(*[Dot([*p, 0], radius=0.06, color=GREY_C1) for p in P])
        params = MathTex(r"\varepsilon = %.2f,\quad \mathrm{MinPts}=%d" % (eps, minpts), color=INK, font_size=32)
        params.to_corner(UR, buff=0.4).shift(DOWN * 0.7)
        self.play(FadeIn(dots), Write(params))

        # point types
        border = (~core) & np.array([core[nb[i]].any() for i in range(len(P))])
        noisep = (~core) & (~border)
        counts = np.array([len(x) for x in nb])
        ic = int(np.argmax(np.where(core, counts, -1)))
        ib = int(np.where(border)[0][np.argmin(D[np.where(border)[0]][:, ic])]) if border.any() else ic
        inn = int(np.where(noisep)[0][0])
        def pl(n):
            return f"{n} point{'s' if n != 1 else ''}"
        demo = [(ic, GREEN_C1, f"{pl(counts[ic])} within ε (≥ MinPts)  →  core point"),
                (ib, ORANGE_C1, f"{pl(counts[ib])} within ε, but one is a core point  →  border point"),
                (inn, RED_C1, f"{pl(counts[inn])} within ε, no core point nearby  →  noise")]
        for i, col, txt in demo:
            c = Circle(radius=eps, color=col, stroke_width=3).move_to(dots[i].get_center())
            c.set_fill(col, opacity=0.08)
            inside = VGroup(*[dots[j] for j in nb[i]])
            self.say(txt)
            self.play(Create(c), inside.animate.set_color(INK), dots[i].animate.set_color(col).scale(1.6), run_time=0.8)
            self.wait(0.9)
            self.play(FadeOut(c), inside.animate.set_color(GREY_C1), dots[i].animate.set_color(GREY_C1).scale(1 / 1.6),
                      run_time=0.4)

        # flood fill
        lab = -np.ones(len(P), int)
        cid = 0
        for seed in np.argsort(-counts):
            if not core[seed] or lab[seed] >= 0:
                continue
            col = PAL[cid]
            self.say(f"Cluster {cid + 1}: start at a core point, spread through core points within ε.")
            lab[seed] = cid
            frontier = [seed]
            self.play(dots[seed].animate.set_color(col).scale(1.4), run_time=0.3)
            while frontier:
                nxt, wave = [], []
                for i in frontier:
                    for j in nb[i]:
                        if lab[j] < 0:
                            lab[j] = cid
                            wave.append(j)
                            if core[j]:
                                nxt.append(j)
                if wave:
                    self.play(*[dots[j].animate.set_color(col) for j in wave], run_time=0.18)
                frontier = nxt
            cid += 1
        self.say("Border points join their core point's cluster. Isolated points stay noise.")
        xs = VGroup(*[Cross(scale_factor=0.09, stroke_color=GREY_C1, stroke_width=3).move_to(dots[i].get_center())
                      for i in np.where(lab < 0)[0]])
        self.play(FadeIn(xs), *[dots[i].animate.set_opacity(0.0) for i in np.where(lab < 0)[0]])
        self.say(f"{cid} clusters of any shape, number not chosen in advance, noise rejected.")
        self.wait(2)


# ---------------------------------------------------------------- 5. Pothole pipeline
class PotholePipeline(BaseScene):
    def construct(self):
        rng = np.random.default_rng(11)
        self.add(title("Pothole detection = regression + clustering"))
        x = np.sort(rng.uniform(-6, 6, 170))
        road = 0.06 * x - 0.4
        dip = -1.1 * np.exp(-((x + 2.6) / 0.55) ** 2) - 0.8 * np.exp(-((x - 2.9) / 0.4) ** 2)
        z = road + dip + rng.normal(0, 0.08, x.size)
        dots = VGroup(*[Dot([a, b, 0], radius=0.055, color=GREY_C1) for a, b in zip(x, z)])
        side = Text("road scan, side view (depth exaggerated)", color=GREY_C1, font_size=22).move_to([0, 1.6, 0])
        self.play(FadeIn(dots), FadeIn(side))
        self.say("1. Fit the road surface (robustly, so the holes don't pull it down).")
        # robust fit: iterate least squares on points close to the current fit
        keep = np.ones_like(x, bool)
        for _ in range(5):
            a, b = np.polyfit(x[keep], z[keep], 1)
            keep = np.abs(z - (a * x + b)) < 0.25
        fit = Line([-6.4, a * -6.4 + b, 0], [6.4, a * 6.4 + b, 0], color=BLUE_C1, stroke_width=4)
        self.play(Create(fit))
        self.say("2. Residual = height of each point above or below the fitted surface.")
        res = z - (a * x + b)
        segs = VGroup(*[Line([xi, zi, 0], [xi, a * xi + b, 0], color=BLUE_C1, stroke_width=1.5, stroke_opacity=0.6)
                        for xi, zi in zip(x, z)])
        self.play(Create(segs), run_time=1.2)
        thr = -0.3
        tl = DashedLine([-6.4, a * -6.4 + b + thr, 0], [6.4, a * 6.4 + b + thr, 0], color=ORANGE_C1, stroke_width=3)
        tlab = Text("threshold ≈ 3σ of the noise", color=ORANGE_C1, font_size=19).next_to(tl, DOWN, buff=0.15).to_edge(LEFT, buff=0.25)
        self.say("3. Keep only points clearly below the surface (beyond the sensor noise).")
        self.play(FadeOut(segs), Create(tl), FadeIn(tlab))
        below = np.where(res < thr)[0]
        self.play(*[dots[i].animate.set_color(ORANGE_C1).scale(1.3) for i in below], run_time=0.8)
        self.say("4. DBSCAN on those points: each cluster is one pothole.")
        # 1D DBSCAN-like grouping by gaps in x
        xb = x[below]
        groups, cur = [], [below[0]]
        for i_prev, i in zip(below[:-1], below[1:]):
            if x[i] - x[i_prev] < 0.5:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        groups = [g for g in groups if len(g) >= 3]
        anims, labels = [], VGroup()
        for k, g in enumerate(groups):
            col = [PURPLE_C1, RED_C1, GREEN_C1][k % 3]
            anims += [dots[i].animate.set_color(col) for i in g]
            br = Brace(VGroup(*[dots[i] for i in g]), DOWN, color=col)
            lb = Text(f"pothole {k + 1}", color=col, font_size=24)
            lb.next_to(br, DOWN, buff=0.1)
            labels.add(VGroup(br, lb))
        self.play(*anims, run_time=0.8)
        self.play(FadeIn(labels))
        self.say("Number of potholes found automatically; isolated noisy dips are rejected.")
        self.wait(2.2)
