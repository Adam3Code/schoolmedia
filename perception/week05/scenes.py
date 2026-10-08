"""Manim scenes for Perception for Autonomous Systems, Week 5 (Point Cloud Processing & Pose Estimation).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
Shown in 2D so every step is visible; the 3D algorithms are identical with 3x3 matrices.
"""
from manim import *
import numpy as np

Text.set_default(font="Inter")

BLUE_C1, RED_C1, GREEN_C1, PURPLE_C1, ORANGE_C1 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"
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


# ---------------------------------------------------------------- shared geometry
def polyline(pts, n):
    pts = np.array(pts, float)
    seg = np.diff(pts, axis=0)
    L = np.linalg.norm(seg, axis=1)
    c = np.r_[0, np.cumsum(L)]
    out = []
    for v in np.linspace(0, c[-1], n):
        i = min(np.searchsorted(c, v, side="right") - 1, len(seg) - 1)
        out.append(pts[i] + (v - c[i]) / L[i] * seg[i])
    return np.array(out)


SHAPE = polyline([(-2.2, 1.6), (-2.2, -1.4), (1.8, -1.4), (1.8, -0.6), (-1.0, -0.6), (-1.0, 0.9), (-0.3, 1.6)], 60)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, d]) @ U.T
    return Rm, cq - Rm @ cp


def p3(p):
    return np.array([p[0], p[1], 0.0])


# ---------------------------------------------------------------- 1. ICP
class ICPIterations(BaseScene):
    def construct(self):
        rng = np.random.default_rng(2)
        self.add(title("ICP: pair nearest points, solve, move, repeat"))
        S, OFF = 1.15, np.array([-0.8, -0.3])
        tgt = SHAPE * S + OFF
        readout_lab = Text("mean pair distance =", color=INK, font_size=24).to_corner(UR, buff=0.5).shift(DOWN * 0.7 + LEFT * 0.9)
        err = ValueTracker(0)
        readout = always_redraw(lambda: DecimalNumber(err.get_value(), num_decimal_places=2, color=INK, font_size=26)
                                .next_to(readout_lab, RIGHT, buff=0.15))
        legend = VGroup(
            VGroup(Dot(color=BLUE_C1), Text("model 2 (fixed)", color=INK, font_size=20)).arrange(RIGHT, buff=0.15),
            VGroup(Dot(color=RED_C1), Text("model 1 (moves)", color=INK, font_size=20)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.5).shift(DOWN * 0.7)

        def run(ang, shift, n_iter, slow_first=2):
            src = (SHAPE + rng.normal(0, 0.03, SHAPE.shape)) @ rot(np.radians(ang)).T * S + OFF + np.array(shift)
            sdots = VGroup(*[Dot(p3(p), radius=0.055, color=RED_C1) for p in src])
            self.play(FadeIn(sdots))
            for k in range(n_iter):
                nn = np.argmin(((src[:, None] - tgt[None]) ** 2).sum(-1), 1)
                Q = tgt[nn]
                e = float(np.mean(np.linalg.norm(src - Q, axis=1)))
                lines = VGroup(*[Line(p3(a), p3(b), color=GREY_C1, stroke_width=1.5) for a, b in zip(src, Q)])
                Rm, t = kabsch(src, Q)
                new = src @ Rm.T + t
                slow = k < slow_first
                if k == slow_first:
                    self.say("Repeat: re-pair with the new positions, re-solve, move.")
                if slow:
                    self.say(f"Iteration {k + 1}, step 1: pair each red point with its nearest blue point.")
                self.play(Create(lines), err.animate.set_value(e), run_time=0.8 if slow else 0.3)
                if slow:
                    self.say(f"Iteration {k + 1}, step 2: Kabsch finds the best R, t for these pairs; move.")
                self.play(*[d.animate.move_to(p3(p)) for d, p in zip(sdots, new)], FadeOut(lines),
                          run_time=1.0 if slow else 0.45)
                src = new
            e = float(np.mean(np.min(np.linalg.norm(src[:, None] - tgt[None], axis=2), axis=1)))
            self.play(err.animate.set_value(e), run_time=0.3)
            return sdots

        self.play(LaggedStart(*[FadeIn(Dot(p3(p), radius=0.055, color=BLUE_C1)) for p in tgt], lag_ratio=0.01),
                  FadeIn(legend), FadeIn(readout_lab))
        self.add(readout)
        self.say("Start close: model 1 is off by about 30°.")
        sd = run(28, (1.1, 0.7), 18)
        self.say("Pairs get better every round, so the error keeps falling: aligned.")
        self.wait(1.4)
        self.play(FadeOut(sd))
        self.say("Now start far off: rotated by 150°.")
        sd = run(150, (0.9, 0.5), 9, slow_first=1)
        self.say("Stuck: most nearest neighbours were wrong from the start. A local minimum.")
        self.wait(1.2)
        self.say("That's why a global method (descriptors + RANSAC) gives ICP its starting pose.")
        self.wait(2)


# ---------------------------------------------------------------- 2. Normals
class NormalEstimation(BaseScene):
    def construct(self):
        rng = np.random.default_rng(5)
        self.add(title("Normals: the direction the neighbours don't spread"))
        floor = np.c_[np.linspace(-5.0, 2.4, 70), np.full(70, -1.6)]
        wall = np.c_[np.full(32, 2.4), np.linspace(-1.5, 1.7, 32)]
        P = np.vstack([floor, wall]) + rng.normal(0, 0.035, (102, 2))
        dots = VGroup(*[Dot(p3(p), radius=0.05, color=GREY_C1) for p in P])
        sensor = Triangle(color=INK, fill_color=INK, fill_opacity=1).scale(0.18).rotate(PI).move_to([-1.5, 2.3, 0])
        slab = Text("sensor", color=INK, font_size=20).next_to(sensor, RIGHT, buff=0.15)
        self.play(FadeIn(dots), FadeIn(sensor), FadeIn(slab))

        def pca(i, r):
            nb = np.where(np.linalg.norm(P - P[i], axis=1) <= r)[0]
            X = P[nb]
            c = X.mean(0)
            C = (X - c).T @ (X - c) / len(X)
            w, V = np.linalg.eigh(C)  # ascending
            n = V[:, 0]
            if n @ (np.array([-1.5, 2.3]) - P[i]) < 0:
                n = -n
            return nb, c, w, V, n

        def show(i, r, col, long=True, wrong=False):
            nb, c, w, V, n = pca(i, r)
            circ = Circle(radius=r, color=PURPLE_C1, stroke_width=2.5).move_to(p3(P[i]))
            hi = VGroup(*[dots[j] for j in nb])
            ang = np.arctan2(V[1, 1], V[0, 1])
            ell = Ellipse(width=max(4 * np.sqrt(w[1]), 0.05), height=max(4 * np.sqrt(w[0]), 0.05),
                          color=ORANGE_C1, stroke_width=3).rotate(ang).move_to(p3(c))
            arr = Arrow(p3(P[i]), p3(P[i] + 0.9 * n), buff=0, color=col, stroke_width=6,
                        max_tip_length_to_length_ratio=0.25)
            if long:
                self.play(Create(circ), hi.animate.set_color(INK), run_time=0.8)
                self.say("Fit an ellipse to the neighbours (their covariance): long along the surface, thin across.")
                self.play(Create(ell))
                self.wait(0.6)
                self.say("Normal = smallest-eigenvalue direction, flipped to face the sensor.")
                self.play(GrowArrow(arr))
                self.wait(0.8)
            else:
                self.play(Create(circ), hi.animate.set_color(INK), Create(ell), GrowArrow(arr), run_time=0.7)
                if wrong:
                    self.wait(1.6)
            self.play(FadeOut(circ), FadeOut(ell), hi.animate.set_color(GREY_C1), run_time=0.4)
            return arr

        i0 = 28
        self.say("Pick a point and take its neighbours within radius r.")
        arrows = [show(i0, 0.7, GREEN_C1)]
        self.say("Same at every point: floor normals point up, wall normals point into the room.")
        for i in [8, 50, 62, 80, 92]:
            arrows.append(show(i, 0.7, GREEN_C1, long=False))
        self.wait(0.8)
        self.say("Radius too big at a corner: the neighbours span two surfaces.")
        icorner = int(np.argmin(np.linalg.norm(P - [2.0, -1.5], axis=1)))
        arrows.append(show(icorner, 1.9, RED_C1, long=False, wrong=True))
        self.say("The ellipse tilts, so the normal points diagonally. Use r ≈ 2–3 × point spacing.")
        self.wait(2)


# ---------------------------------------------------------------- 3. RANSAC
class RANSACMatches(BaseScene):
    def construct(self):
        rng = np.random.default_rng(8)
        self.add(title("RANSAC: find the pose most matches agree on"))
        S = 0.85
        A = SHAPE * S + np.array([-3.6, -0.2])
        Rtrue = rot(np.radians(55))
        B = (SHAPE * S) @ Rtrue.T + np.array([3.4, -0.3])
        da = VGroup(*[Dot(p3(p), radius=0.045, color=BLUE_C1) for p in A])
        db = VGroup(*[Dot(p3(p), radius=0.045, color=RED_C1) for p in B])
        self.play(FadeIn(da), FadeIn(db))

        kp = np.array([2, 7, 12, 17, 22, 27, 31, 36, 40, 44, 48, 52, 56, 59])
        n = len(kp)
        good = np.zeros(n, bool)
        good[[0, 2, 3, 5, 7, 9, 10, 12]] = True
        partner = kp.copy()
        for j in np.where(~good)[0]:
            choices = [k for k in range(60) if abs(k - kp[j]) > 10]
            partner[j] = rng.choice(choices)
        lines = VGroup(*[Line(p3(A[a]), p3(B[b]), color=GREY_C1, stroke_width=1.6) for a, b in zip(kp, partner)])
        self.say(f"{n} feature matches from FPFH: some right, some wrong. RANSAC can't tell which.")
        self.play(Create(lines), run_time=1.2)
        self.wait(0.8)

        lab = Text("inliers:", color=INK, font_size=24).to_corner(UR, buff=0.5).shift(DOWN * 0.7 + LEFT * 1.4)
        best_lab = Text("best:", color=INK, font_size=24).next_to(lab, DOWN, buff=0.2).align_to(lab, LEFT)
        cur = ValueTracker(0)
        best = ValueTracker(0)
        curn = always_redraw(lambda: Text(f"{int(round(cur.get_value()))} / {n}", color=INK, font_size=24).next_to(lab, RIGHT, buff=0.15))
        bestn = always_redraw(lambda: Text(f"{int(round(best.get_value()))} / {n}", color=GREEN_C1, font_size=24).next_to(best_lab, RIGHT, buff=0.15))
        self.play(FadeIn(lab), FadeIn(best_lab))
        self.add(curn, bestn)
        thr = 0.25
        samples = [[1, 4, 9], [0, 6, 11], [3, 7, 10], [2, 8, 12]]
        best_inl = None
        for it, smp in enumerate(samples):
            Rm, t = kabsch(A[kp[smp]], B[partner[smp]])
            moved = A @ Rm.T + t
            d = np.linalg.norm(A[kp] @ Rm.T + t - B[partner], axis=1)
            inl = d < thr
            slow = it == 0
            if slow:
                self.say("Iteration 1: pick 3 random matches and solve R, t from them (Kabsch).")
            hl = VGroup(*[lines[j].copy().set_color(ORANGE_C1).set_stroke(width=5) for j in smp])
            self.play(Create(hl), run_time=0.6 if slow else 0.35)
            ghost = VGroup(*[Dot(p3(p), radius=0.05, color=PURPLE_C1, fill_opacity=0.85) for p in moved])
            if slow:
                self.say("Move the blue model with that pose (purple) and count matches that land on their partner.")
            self.play(FadeIn(ghost), cur.animate.set_value(int(inl.sum())), run_time=0.8 if slow else 0.5)
            if inl.sum() > best.get_value():
                self.play(best.animate.set_value(int(inl.sum())), run_time=0.3)
                best_inl = inl
            if slow:
                self.say("A wrong match was in the sample, so the pose is wrong and few matches agree.")
                self.wait(1.0)
            elif it == len(samples) - 2:
                self.say("All 3 sampled matches correct → the pose fits and many matches agree.")
                self.wait(1.2)
            else:
                self.wait(0.5)
            self.play(FadeOut(ghost), FadeOut(hl), run_time=0.3)
            if it == 0:
                self.say("Repeat with new random samples; keep the pose with the most inliers.")
        self.say("Keep the best: its inliers are the right matches. Refit → rough pose for ICP.")
        self.play(cur.animate.set_value(best.get_value()), run_time=0.3)
        self.play(*[lines[j].animate.set_color(GREEN_C1 if best_inl[j] else RED_C1).set_stroke(width=2.5) for j in range(n)])
        Rm, t = kabsch(A[kp[best_inl]], B[partner[best_inl]])
        final = VGroup(*[Dot(p3(p), radius=0.045, color=BLUE_C1) for p in A @ Rm.T + t])
        self.play(TransformFromCopy(da, final), run_time=1.4)
        eq = MathTex(r"N=\frac{\log(1-P)}{\log(1-w^3)}", color=INK, font_size=34).to_corner(UL, buff=0.5).shift(DOWN * 0.7)
        self.play(Write(eq))
        self.say("Fewer good matches → many more tries: w = 0.5 needs 35, w = 0.2 needs 574.")
        self.wait(2.2)
