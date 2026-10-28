"""Manim scenes for Perception for Autonomous Systems, Week 8 (Classification).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
All scenes compute the real algorithm with NumPy, so the motion shows what the method actually does.
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


# ---------------------------------------------------------------- 1. PCA
class PCARotatingAxis(BaseScene):
    def construct(self):
        rng = np.random.default_rng(3)
        self.add(title("PCA: find the direction with the most spread"))
        n = 40
        t = rng.normal(0, 1.25, n)
        s = rng.normal(0, 0.45, n)
        ang0 = np.deg2rad(28)
        R = np.array([[np.cos(ang0), -np.sin(ang0)], [np.sin(ang0), np.cos(ang0)]])
        P = (R @ np.vstack([t, s])).T + np.array([0.3, -0.2])
        c = P.mean(0)
        X = P - c
        C = X.T @ X / (n - 1)
        evals, evecs = np.linalg.eigh(C)
        e1, e2 = evecs[:, 1], evecs[:, 0]
        best = np.arctan2(e1[1], e1[0])

        dots = VGroup(*[Dot([*p, 0], radius=0.06, color=GREY_C1) for p in P])
        th = ValueTracker(np.deg2rad(-50))

        def u():
            a = th.get_value()
            return np.array([np.cos(a), np.sin(a)])

        def axis_line():
            d = u()
            return Line([*(c - 4.2 * d), 0], [*(c + 4.2 * d), 0], color=RED_C1, stroke_width=4)

        def shadows():
            d = u()
            pr = X @ d
            return VGroup(*[Dot([*(c + v * d), 0], radius=0.045, color=RED_C1) for v in pr])

        def var():
            return float(np.var(X @ u(), ddof=1))

        ax = always_redraw(axis_line)
        sh = always_redraw(shadows)
        lab = Text("variance along line =", color=INK, font_size=26)
        lab.to_corner(UL, buff=0.5).shift(DOWN * 0.75)
        val = always_redraw(lambda: DecimalNumber(var(), num_decimal_places=2, color=RED_C1, font_size=30)
                            .next_to(lab, RIGHT, buff=0.15))

        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.03), run_time=1.2)
        self.say("Each dot is a sample with 2 features. Which single direction describes it best?")
        self.wait(0.6)
        self.say("Drop every dot onto a line (red shadows) and measure how spread out they are.")
        self.add(ax, sh, lab, val)
        self.wait(0.8)
        self.say("Turn the line: the spread of the shadows changes.")
        self.play(th.animate.set_value(np.deg2rad(110)), run_time=3.2, rate_func=there_and_back_with_pause)
        self.say("The direction with the largest spread is the 1st principal component.")
        self.play(th.animate.set_value(best), run_time=1.6)
        self.wait(0.6)
        eq = MathTex(r"C\,e_1=\lambda_1 e_1", color=INK, font_size=40).next_to(lab, DOWN, buff=0.4).align_to(lab, LEFT)
        lam = MathTex(rf"\lambda_1={evals[1]:.2f},\ \ \lambda_2={evals[0]:.2f}", color=INK, font_size=34)
        lam.next_to(eq, DOWN, buff=0.25).align_to(eq, LEFT)
        self.say("It is the eigenvector of the covariance with the biggest eigenvalue.")
        self.play(Write(eq), Write(lam))
        self.wait(1.0)
        pc2 = Arrow([*c, 0], [*(c + 1.6 * e2), 0], buff=0, color=BLUE_C1, stroke_width=5)
        self.say("PC2 is at a right angle to it and holds the little spread that is left.")
        self.play(GrowArrow(pc2))
        self.wait(0.8)
        pct = evals[1] / evals.sum() * 100
        self.say(f"PC1 alone explains {pct:.0f} % of the variance: keep it, drop PC2.")
        targets = [Dot([*(c + (x @ e1) * e1), 0], radius=0.06, color=RED_C1) for x in X]
        self.play(*[Transform(d, tg) for d, tg in zip(dots, targets)], FadeOut(pc2), run_time=1.8)
        self.wait(0.6)
        self.say("Every sample is now one number y = e₁ᵀ(x − mean): 2D became 1D.")
        self.wait(1.6)


# ---------------------------------------------------------------- 2. k-NN
class KNNVote(BaseScene):
    def construct(self):
        rng = np.random.default_rng(11)
        self.add(title("k-NN: ask the k nearest neighbours"))
        red = rng.normal([-1.8, 0.2], 0.9, (16, 2))
        blue = rng.normal([2.0, -0.2], 0.9, (16, 2))
        q = np.array([-0.7, 0.3])
        stray = np.array([-0.55, 0.05])           # one noisy blue sample next to the query
        blue = np.vstack([blue, stray])
        P = np.vstack([red, blue])
        lab = np.r_[np.zeros(len(red), int), np.ones(len(blue), int)]
        col = [RED_C1, BLUE_C1]
        dots = VGroup(*[Dot([*p, 0], radius=0.08, color=col[l]) for p, l in zip(P, lab)])
        qd = Dot([*q, 0], radius=0.11, color=INK)
        ql = Text("?", color=INK, font_size=26).next_to(qd, UP, buff=0.1)
        d = np.linalg.norm(P - q, axis=1)
        order = np.argsort(d)

        self.play(LaggedStart(*[FadeIn(x, scale=0.5) for x in dots], lag_ratio=0.03), run_time=1.2)
        self.say("Two classes of training samples. A new sample (black) arrives.")
        self.play(FadeIn(qd, scale=0.5), FadeIn(ql))
        self.wait(0.4)
        self.say("Measure the distance to every training sample, keep the k closest.")

        readout = VGroup()
        prev = VGroup()
        for k in [1, 5, 9]:
            idx = order[:k]
            r = d[idx[-1]] + 0.08
            circ = DashedVMobject(Circle(radius=r, color=INK, stroke_width=2).move_to([*q, 0]), num_dashes=40)
            links = VGroup(*[Line([*q, 0], [*P[i], 0], color=col[lab[i]], stroke_width=2) for i in idx])
            kr = int(np.sum(lab[idx] == 0))
            winner = "red" if kr > k - kr else "blue"
            txt = VGroup(
                Text(f"k = {k}", color=INK, font_size=28, weight=BOLD),
                MathTex(rf"p(\text{{red}})=\tfrac{{{kr}}}{{{k}}}", color=RED_C1, font_size=34),
                Text(f"→ {winner}", color=col[0 if winner == 'red' else 1], font_size=28),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UL, buff=0.5).shift(DOWN * 0.8)
            self.play(FadeOut(prev), FadeOut(readout), Create(circ), Create(links), run_time=1.0)
            self.play(FadeIn(txt))
            if k == 1:
                self.say("k = 1: one noisy blue neighbour decides alone → blue. Fragile!")
            elif k == 5:
                self.say("k = 5: four red, one blue → red with probability 4/5.")
            else:
                self.say("k = 9: a bigger vote smooths out the noise, still red.")
            self.wait(1.4)
            prev = VGroup(circ, links)
            readout = txt
        self.say("Small k follows noise; very large k blurs the boundary. Pick k by validation.")
        self.wait(1.6)


# ---------------------------------------------------------------- 3. SVM
class SVMMaxMargin(BaseScene):
    def construct(self):
        rng = np.random.default_rng(5)
        self.add(title("SVM: the widest street between the classes"))
        neg = rng.normal([-1.9, -0.9], 0.55, (12, 2))
        pos = rng.normal([1.7, 0.9], 0.55, (12, 2))
        allp = np.vstack([neg, pos])
        cnt = allp.mean(0)

        def gap(a):
            n = np.array([np.cos(a), np.sin(a)])
            return (pos @ n).min() - (neg @ n).max(), n

        angs = np.linspace(-np.pi / 2, np.pi / 2, 721)
        gaps = np.array([gap(a)[0] for a in angs])
        a_best = angs[np.argmax(gaps)]

        th = ValueTracker(np.deg2rad(-5))

        def street():
            g, n = gap(th.get_value())
            lo = (neg @ n).max()
            hi = (pos @ n).min()
            mid = (lo + hi) / 2
            tdir = np.array([-n[1], n[0]])
            base = cnt - (cnt @ n) * n
            grp = VGroup()
            if g > 0:
                quad = Polygon(*[[*(base + lo * n + s * tdir), 0] for s in (-3.0, 3.0)],
                               *[[*(base + hi * n + s * tdir), 0] for s in (3.0, -3.0)],
                               stroke_width=0, fill_color=GREEN_C1, fill_opacity=0.15)
                grp.add(quad)
            for off, w, dash in [(lo, 2, True), (hi, 2, True), (mid, 4, False)]:
                L = Line([*(base + off * n - 3.0 * tdir), 0], [*(base + off * n + 3.0 * tdir), 0],
                         color=INK if dash else GREEN_C1, stroke_width=w)
                grp.add(DashedVMobject(L, num_dashes=26) if dash else L)
            return grp

        def width():
            return max(gap(th.get_value())[0], 0.0)

        nd = VGroup(*[Dot([*p, 0], radius=0.08, color=RED_C1) for p in neg])
        pd = VGroup(*[Dot([*p, 0], radius=0.08, color=BLUE_C1) for p in pos])
        self.play(LaggedStart(*[FadeIn(x, scale=0.5) for x in [*nd, *pd]], lag_ratio=0.03), run_time=1.2)
        self.say("Many lines separate the two classes. Which one should we trust?")
        st = always_redraw(street)
        lab = Text("street width =", color=INK, font_size=26).to_corner(UL, buff=0.5).shift(DOWN * 0.75)
        val = always_redraw(lambda: DecimalNumber(width(), num_decimal_places=2, color=GREEN_C1, font_size=30)
                            .next_to(lab, RIGHT, buff=0.15))
        self.add(st, lab, val)
        self.bring_to_front(nd, pd)
        self.say("For each direction, make the street as wide as the points allow.")
        self.play(th.animate.set_value(np.deg2rad(75)), run_time=2.4)
        self.say("Tilted this way, points nearly touch it: the street is narrow.")
        self.wait(0.6)
        self.say("SVM picks the direction with the widest street: the max margin.")
        self.play(th.animate.set_value(a_best), run_time=2.0)
        self.wait(0.4)
        g, n = gap(a_best)
        lo, hi = (neg @ n).max(), (pos @ n).min()
        sv = [p for p in neg if lo - p @ n < 0.03] + [p for p in pos if p @ n - hi < 0.03]
        rings = VGroup(*[Circle(radius=0.2, color=ORANGE_C1, stroke_width=4).move_to([*p, 0]) for p in sv])
        self.say("Only the points on the kerbs fix the street: the support vectors.")
        self.play(Create(rings))
        self.wait(1.0)
        eq = MathTex(r"\text{margin}=\frac{2}{\|w\|}", color=INK, font_size=38)
        eq.next_to(lab, DOWN, buff=0.4).align_to(lab, LEFT)
        rule = MathTex(r"f(x)=\operatorname{sgn}(w^\top x+b)", color=INK, font_size=34)
        rule.next_to(eq, DOWN, buff=0.25).align_to(eq, LEFT)
        self.say("Wide street = small ‖w‖. New sample: which side of the centre line?")
        self.play(Write(eq), Write(rule))
        self.wait(2.0)


# ---------------------------------------------------------------- 4. Kernel trick
class KernelTrick(BaseScene):
    def construct(self):
        self.add(title("Kernel trick: lift the data until a line works"))
        xs_neg = np.array([-1.2, -0.6, 0.0, 0.5, 1.1])
        xs_pos = np.array([-2.7, -2.2, 2.1, 2.6, 3.0])
        sx, sy = 1.3, 0.42           # screen scaling for x and x²
        y0 = -2.0

        axis = Line([-5.5, y0, 0], [5.5, y0, 0], color=INK, stroke_width=2)
        xl = MathTex("x", color=INK, font_size=32).next_to(axis, RIGHT, buff=0.1)
        neg = VGroup(*[Dot([x * sx, y0, 0], radius=0.09, color=RED_C1) for x in xs_neg])
        pos = VGroup(*[Dot([x * sx, y0, 0], radius=0.09, color=BLUE_C1) for x in xs_pos])
        self.play(Create(axis), FadeIn(xl))
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in [*neg, *pos]], lag_ratio=0.05))
        self.say("1D data: red in the middle, blue on both sides.")
        thr = DashedLine([0.0, y0 - 0.6, 0], [0.0, y0 + 0.6, 0], color=GREEN_C1, stroke_width=4)
        self.play(Create(thr))
        self.play(thr.animate.shift(RIGHT * 2.2), run_time=1.0)
        self.play(thr.animate.shift(LEFT * 4.4), run_time=1.2)
        self.say("No single threshold separates them: not linearly separable.")
        self.play(FadeOut(thr))
        self.wait(0.4)

        self.say("Add a feature: map each x to the point (x, x²).")
        curve = ParametricFunction(lambda t: [t * sx, y0 + sy * t * t, 0], t_range=[-3.3, 3.3],
                                   color=GREY_C1, stroke_width=2)
        yl = MathTex("x^2", color=INK, font_size=32).move_to([-5.0, 1.8, 0])
        self.play(Create(curve), FadeIn(yl))
        self.play(*[d.animate.move_to([x * sx, y0 + sy * x * x, 0]) for d, x in zip(neg, xs_neg)],
                  *[d.animate.move_to([x * sx, y0 + sy * x * x, 0]) for d, x in zip(pos, xs_pos)],
                  run_time=2.0)
        cut = 3.0
        line = DashedLine([-5.2, y0 + sy * cut, 0], [5.2, y0 + sy * cut, 0], color=GREEN_C1, stroke_width=4)
        self.say("In this 2D space a straight line splits them: x² = 3.")
        self.play(Create(line))
        self.wait(1.0)
        eq = MathTex(r"\phi(x)=(x,\,x^2)\qquad k(x,z)=\phi(x)^\top\phi(z)", color=INK, font_size=34)
        eq.to_corner(UR, buff=0.5).shift(DOWN * 0.8)
        self.play(Write(eq))
        self.say("Back in 1D that line becomes two thresholds: |x| = √3 ≈ 1.73.")
        r5 = np.sqrt(cut)
        t1 = DashedLine([-r5 * sx, y0 - 0.5, 0], [-r5 * sx, y0 + sy * cut, 0], color=GREEN_C1, stroke_width=3)
        t2 = DashedLine([r5 * sx, y0 - 0.5, 0], [r5 * sx, y0 + sy * cut, 0], color=GREEN_C1, stroke_width=3)
        self.play(Create(t1), Create(t2))
        self.wait(0.8)
        self.say("A kernel does this lift implicitly, e.g. RBF into infinitely many features.")
        self.wait(2.0)
