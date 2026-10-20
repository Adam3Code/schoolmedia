"""Manim scenes for Perception for Autonomous Systems, Week 7 (State Estimation: Histogram and Kalman Filters).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
All scenes compute the real filter with NumPy, so every number on screen is what the filter actually produces.
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


def gauss(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


# ---------------------------------------------------------------- 1. Histogram filter
class HistogramFilter(BaseScene):
    def construct(self):
        self.add(title("Histogram filter: sense, normalise, move"))
        world = ["g", "r", "r", "g", "g"]
        W = 1.3
        xs = np.array([(i - 2) * W for i in range(5)])
        cells = VGroup(*[Square(W, stroke_color=INK, stroke_width=2,
                                fill_color=GREEN_C1 if c == "g" else RED_C1, fill_opacity=0.35)
                         .move_to([x, 2.1, 0]) for x, c in zip(xs, world)])
        base_y, scale = -2.3, 5.6
        axis = Line([xs[0] - 0.8, base_y, 0], [xs[-1] + 0.8, base_y, 0], color=INK, stroke_width=2)
        idx = VGroup(*[Text(str(i), color=GREY_C1, font_size=20).move_to([x, base_y - 0.25, 0]) for i, x in enumerate(xs)])

        def rects(p, col):
            g = VGroup()
            for x, v in zip(xs, p):
                h = max(v * scale, 0.02)
                g.add(Rectangle(width=0.8, height=h, stroke_width=0, fill_color=col, fill_opacity=0.9)
                      .move_to([x, base_y + h / 2, 0]))
            return g

        def labels(p, rs):
            return VGroup(*[Text(f"{v:.2f}", color=INK, font_size=20).next_to(r, UP, buff=0.08) for v, r in zip(p, rs)])

        state = {}

        def show(p, col=BLUE_C1, run_time=1.0):
            rs = rects(p, col)
            ls = labels(p, rs)
            if not state:
                self.play(FadeIn(rs), FadeIn(ls))
            else:
                self.play(Transform(state["r"], rs), FadeOut(state["l"]), FadeIn(ls), run_time=run_time)
            state["l"] = ls
            state.setdefault("r", rs)

        p = np.full(5, 0.2)
        robot = Triangle(color=PURPLE_C1, fill_opacity=1).scale(0.18).move_to([xs[2], 1.15, 0])
        self.play(FadeIn(cells), Create(axis), FadeIn(idx))
        self.say("No information yet: uniform belief, 0.2 per cell (maximum confusion).")
        show(p)
        self.wait(0.6)

        # sense red
        self.play(FadeIn(robot))
        z = Text("sees: red", color=RED_C1, font_size=24).next_to(robot, RIGHT, buff=0.2)
        self.play(FadeIn(z))
        lik = np.array([0.6 if c == "r" else 0.2 for c in world])
        liks = VGroup(*[Text(f"× {l}", color=INK, font_size=20).move_to([x, 0.7, 0]) for x, l in zip(xs, lik)])
        self.say("Sense: multiply each cell by P(red | cell): 0.6 if red, 0.2 if green.")
        self.play(FadeIn(liks))
        w = p * lik
        show(w, ORANGE_C1)
        s = Text(f"sum = {w.sum():.2f}", color=ORANGE_C1, font_size=26).move_to([xs[-1] + 1.9, -0.6, 0])
        self.say("These weights sum to 0.36, not 1. Divide each by 0.36 (Bayes' denominator).")
        self.play(FadeIn(s))
        p = w / w.sum()
        self.play(FadeOut(s), run_time=0.3)
        show(p)
        self.say("Posterior: the red cells are three times as likely as the green ones.")
        self.wait(0.8)
        self.play(FadeOut(liks), FadeOut(z))

        # move right by 1 with 0.1 / 0.8 / 0.1
        self.say("Move one cell right: 80 % exact, 10 % one short, 10 % one too far (wraps around).")
        src = 2
        arrs = VGroup(*[CurvedArrow([xs[src], 1.35, 0], [xs[(src + d) % 5], 1.35, 0], angle=-PI / 2.2,
                                    color=PURPLE_C1, stroke_width=sw, tip_length=0.18)
                        for d, sw in [(1, 5), (2, 2)]])
        lab = VGroup(Text("0.8", color=PURPLE_C1, font_size=20).move_to([(xs[2] + xs[3]) / 2, 0.55, 0]),
                     Text("0.1", color=PURPLE_C1, font_size=20).move_to([xs[4] - 0.3, 0.4, 0]))
        self.play(Create(arrs), FadeIn(lab))
        self.play(robot.animate.move_to([xs[3], 1.15, 0]))
        q = np.array([0.8 * p[(i - 1) % 5] + 0.1 * p[i % 5] + 0.1 * p[(i - 2) % 5] for i in range(5)])
        show(q, PURPLE_C1, 1.2)
        self.say("Prediction = shift + blur (a convolution). Motion only ever loses certainty.")
        self.wait(0.8)
        self.play(FadeOut(arrs), FadeOut(lab))

        # sense green
        z = Text("sees: green", color=GREEN_C1, font_size=24).next_to(robot, RIGHT, buff=0.2)
        self.play(FadeIn(z))
        lik = np.array([0.6 if c == "g" else 0.2 for c in world])
        w = q * lik
        p = w / w.sum()
        self.say("Sense again (green) and normalise: one cell now clearly wins.")
        show(p, BLUE_C1, 1.2)
        k = int(np.argmax(p))
        win = SurroundingRectangle(VGroup(state["r"][k], state["l"][k]), color=RED_C1, buff=0.08)
        self.play(Create(win), Indicate(cells[k], color=RED_C1))
        self.say(f"Cell {k}: p = {p.max():.2f}. 'Red, then green one step later' fits best there.")
        self.wait(2)


# ---------------------------------------------------------------- 2. Gaussian fusion and prediction
class GaussianFusion(BaseScene):
    def construct(self):
        self.add(title("Kalman in 1D: multiply to update, add to predict"))
        ax = Axes(x_range=[0, 28, 2], y_range=[0, 0.36, 0.1], x_length=11.5, y_length=3.6,
                  axis_config={"color": INK, "stroke_width": 2, "include_ticks": True},
                  x_axis_config={"numbers_to_include": range(0, 29, 4), "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 0, "color": INK}},
                  y_axis_config={"include_tip": False}).shift(DOWN * 0.8)
        self.play(Create(ax))

        def curve(mu, var, col, w=4):
            return ax.plot(lambda x: gauss(x, mu, var), x_range=[0, 28, 0.05], color=col, stroke_width=w)

        prior = curve(10, 8, BLUE_C1)
        meas = curve(13, 2, RED_C1)
        lp = Text("prior μ=10, σ²=8", color=BLUE_C1, font_size=22).move_to(ax.c2p(4.5, 0.17))
        lm = Text("measurement ν=13, r²=2", color=RED_C1, font_size=22).move_to(ax.c2p(19.5, 0.3))
        self.say("Prior belief (blue) and a more precise measurement (red).")
        self.play(Create(prior), FadeIn(lp))
        self.play(Create(meas), FadeIn(lm))
        eq = MathTex(r"\mu' = \frac{r^2\mu + \sigma^2\nu}{\sigma^2 + r^2},\quad \sigma'^2 = \frac{\sigma^2 r^2}{\sigma^2+r^2}",
                     color=INK, font_size=34).move_to([0, 2.35, 0])
        self.play(Write(eq))
        self.say("Update = multiply the two Gaussians: a weighted mean, weights = the other's variance.")
        post = curve(12.4, 1.6, GREEN_C1, 5)
        lq = Text("posterior μ'=12.4, σ'²=1.6", color=GREEN_C1, font_size=22).move_to(ax.c2p(19.5, 0.36))
        self.play(Create(post), FadeIn(lq), lm.animate.move_to(ax.c2p(19.5, 0.23)))
        self.say("Mean lands nearer the precise one (12.4); variance 1.6 is smaller than BOTH inputs.")
        self.wait(1.6)

        # prediction example from the slides
        self.play(*[FadeOut(m) for m in [prior, meas, post, lp, lm, lq, eq]])
        eq2 = MathTex(r"\mu' = \mu + \nu,\qquad \sigma'^2 = \sigma^2 + r^2", color=INK, font_size=36).move_to([0, 2.35, 0])
        p0 = curve(8, 4, BLUE_C1)
        l0 = Text("before: μ=8, σ²=4", color=BLUE_C1, font_size=22).move_to(ax.c2p(4, 0.26))
        self.say("Prediction: move by ν = 10, with motion noise r² = 6.")
        self.play(Create(p0), FadeIn(l0), Write(eq2))
        ghost = p0.copy().set_stroke(opacity=0.5)
        self.add(ghost)
        p1 = curve(18, 10, PURPLE_C1, 5)
        l1 = Text("predicted: μ=18, σ²=10", color=PURPLE_C1, font_size=22).move_to(ax.c2p(22.5, 0.17))
        arrow = Arrow(ax.c2p(8, 0.21), ax.c2p(18, 0.21), buff=0, color=ORANGE_C1, stroke_width=5)
        self.play(GrowArrow(arrow))
        self.play(ReplacementTransform(ghost, p1), FadeIn(l1), run_time=1.4)
        self.say("Means add, variances add: moving always makes the belief wider.")
        self.wait(1.2)
        self.say("Update shrinks the Gaussian, prediction grows it. The filter alternates the two.")
        self.wait(2)


# ---------------------------------------------------------------- 3. Scalar exercise run
class ScalarKalmanRun(BaseScene):
    def construct(self):
        self.add(title("The scalar filter exercise, step by step"))
        Z, U, R, Q = [5, 6, 7, 9, 10], [1, 1, 2, 1, 1], 4.0, 2.0
        ax = Axes(x_range=[0, 14, 1], y_range=[0, 0.3, 0.1], x_length=7.6, y_length=3.6,
                  axis_config={"color": INK, "stroke_width": 2},
                  x_axis_config={"numbers_to_include": range(0, 15, 2), "font_size": 20,
                                 "decimal_number_config": {"num_decimal_places": 0, "color": INK}},
                  y_axis_config={"include_tip": False}).to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        info = Text("start μ=0, σ²=10000\nR = 4 (sensor)   Q = 2 (motion)", color=INK, font_size=20, line_spacing=0.8)
        info.to_corner(UR, buff=0.4).shift(DOWN * 0.75)
        self.play(Create(ax), FadeIn(info))
        hdr = VGroup(*[Text(t, color=INK, font_size=20, weight=BOLD) for t in ["z", "μ (upd)", "σ² (upd)", "σ² (pred)"]])
        colx = [2.6, 3.6, 4.85, 6.15]
        for t, x in zip(hdr, colx):
            t.move_to([x, 1.25, 0])
        self.play(FadeIn(hdr))

        def curve(mu, var, col):
            return ax.plot(lambda x: gauss(x, mu, var), x_range=[0, 14, 0.04], color=col, stroke_width=4)

        mu, var = 0.0, 10000.0
        cur = curve(mu, var, BLUE_C1)
        self.add(cur)
        self.say("Prior σ² = 10000: almost flat, the filter knows nothing yet.")
        self.wait(0.8)
        rows = VGroup()
        for k, (z, u) in enumerate(zip(Z, U)):
            y = 0.8 - 0.42 * k
            zl = DashedLine(ax.c2p(z, 0), ax.c2p(z, 0.28), color=RED_C1, stroke_width=3)
            if k == 0:
                self.say("Measurement z = 5 with R = 4: the posterior snaps to it.")
            self.play(Create(zl), run_time=0.4)
            mu = (R * mu + var * z) / (var + R)
            var = var * R / (var + R)
            new = curve(mu, var, GREEN_C1)
            self.play(ReplacementTransform(cur, new), run_time=0.7)
            cur = new
            r = VGroup(*[Text(s, color=c, font_size=20) for s, c in
                         [(str(z), RED_C1), (f"{mu:.2f}", GREEN_C1), (f"{var:.2f}", GREEN_C1)]])
            for t, x in zip(r, colx):
                t.move_to([x, y, 0])
            self.play(FadeIn(r), run_time=0.3)
            if k == 0:
                self.say(f"Then predict: move by u = {u}, add Q = 2 to the variance.")
            mu, var = mu + u, var + Q
            new = curve(mu, var, PURPLE_C1)
            pv = Text(f"{var:.2f}", color=PURPLE_C1, font_size=20).move_to([colx[3], y, 0])
            self.play(ReplacementTransform(cur, new), FadeOut(zl), FadeIn(pv), run_time=0.7)
            cur = new
            rows.add(r, pv)
            if k == 1:
                self.say("Notice the variance: it stops changing after a few steps.")
        box = SurroundingRectangle(VGroup(*rows[-4:]), color=ORANGE_C1, buff=0.08)
        self.play(Create(box))
        self.say("Steady state: σ² ≈ 2 after each update, ≈ 4 after each prediction. Final μ ≈ 11.")
        self.wait(2.2)


# ---------------------------------------------------------------- 4. Position-velocity Kalman
class PositionVelocity(BaseScene):
    def construct(self):
        self.add(title("Kalman in 2D: position measurements reveal velocity"))
        ax = Axes(x_range=[0, 5, 1], y_range=[-1.5, 3, 1], x_length=8.0, y_length=4.6,
                  axis_config={"color": INK, "stroke_width": 2},
                  x_axis_config={"numbers_to_include": range(0, 6), "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 0, "color": INK}},
                  y_axis_config={"numbers_to_include": [-1, 0, 1, 2, 3], "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 0, "color": INK}}
                  ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.35)
        xl = Text("position x", color=INK, font_size=20).next_to(ax.x_axis, DOWN, buff=0.35).shift(RIGHT * 3)
        vl = Text("velocity ẋ", color=INK, font_size=20).next_to(ax.y_axis, UP, buff=0.1).shift(RIGHT * 0.6)
        self.play(Create(ax), FadeIn(xl), FadeIn(vl))
        F = np.array([[1, 1], [0, 1.]])
        R = 0.03
        eqs = VGroup(MathTex(r"F=\begin{pmatrix}1&1\\0&1\end{pmatrix}", color=INK, font_size=30),
                     MathTex(r"H=\begin{pmatrix}1&0\end{pmatrix}", color=INK, font_size=30),
                     MathTex(r"P' = FPF^\top", color=INK, font_size=30)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        eqs.to_corner(UR, buff=0.5).shift(DOWN * 0.8)
        self.play(Write(eqs))

        def ell(x, P, col, k=2.0, fill=0.12):
            vals, vecs = np.linalg.eigh(P)
            t = np.linspace(0, 2 * np.pi, 120)
            pts = (vecs @ np.diag(k * np.sqrt(np.maximum(vals, 1e-9))) @ np.vstack([np.cos(t), np.sin(t)])).T + x
            pts[:, 1] = np.clip(pts[:, 1], -1.5, 3.0)
            m = VMobject(stroke_color=col, stroke_width=3, fill_color=col, fill_opacity=fill)
            m.set_points_as_corners([ax.c2p(a, b) for a, b in pts] + [ax.c2p(*pts[0])])
            return VGroup(m, Dot(ax.c2p(*x), radius=0.06, color=col))

        x = np.array([1.0, 0.4])
        P = np.diag([0.03, 0.5])
        e = ell(x, P, BLUE_C1)
        self.say("After z = 1: position known, velocity unknown → a tall, thin belief.")
        self.play(FadeIn(e))
        self.wait(1.0)
        vr = Text("v ≈ 0.40", color=INK, font_size=26).to_corner(UR, buff=0.5).shift(DOWN * 3.3)
        self.play(FadeIn(vr))
        for k, z in enumerate([2, 3, 4]):
            xp, Pp = F @ x, F @ P @ F.T
            ep = ell(xp, Pp, PURPLE_C1)
            if k == 0:
                self.say("Predict: x' = x + ẋ. Fast hypotheses move further, so the belief shears.")
            self.play(ReplacementTransform(e, ep), run_time=1.4)
            e = ep
            if k == 0:
                self.say("Now position and velocity are correlated (tilted ellipse).")
                self.wait(1.0)
            band = Rectangle(width=ax.c2p(z + 2 * np.sqrt(R), 0)[0] - ax.c2p(z - 2 * np.sqrt(R), 0)[0],
                             height=ax.y_length, stroke_width=0, fill_color=GREEN_C1, fill_opacity=0.25)
            band.move_to([ax.c2p(z, 0)[0], ax.c2p(0, 0.75)[1], 0])
            if k == 0:
                self.say("Measure z = 2: only the slice of the ellipse near x = 2 survives.")
            self.play(FadeIn(band), run_time=0.5)
            S = Pp[0, 0] + R
            K = Pp[:, 0] / S
            x = xp + K * (z - xp[0])
            P = Pp - np.outer(K, Pp[0])
            eu = ell(x, P, GREEN_C1, fill=0.25)
            nv = Text(f"v ≈ {x[1]:.2f}", color=INK, font_size=26).move_to(vr)
            self.play(ReplacementTransform(e, eu), FadeOut(band), Transform(vr, nv), run_time=1.1)
            e = eu
            if k == 0:
                self.say("Velocity was never measured, yet it is now ≈ 1: the correlation inferred it.")
                self.wait(1.4)
        self.say("Each predict–update cycle tightens both position and velocity.")
        self.wait(2)
