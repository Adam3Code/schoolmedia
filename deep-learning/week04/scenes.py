"""Manim scenes for 02456 Deep Learning, Week 4 (Convolutional Neural Networks).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
Every number on screen is computed with NumPy in the scene, so the animations show the real operation.
"""
from manim import *
import numpy as np

Text.set_default(font="Inter")

BLUE_C1, RED_C1, GREEN_C1, PURPLE_C1, ORANGE_C1 = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"
GREY_C1 = "#9a9a9a"
INK = "#222222"
LIGHT = "#f2f2f2"


def title(text):
    return Text(text, color=INK, font_size=32, weight=BOLD).to_edge(UP, buff=0.3)


def caption(text):
    return Text(text, color=INK, font_size=22).to_edge(DOWN, buff=0.3)


class BaseScene(Scene):
    def setup(self):
        self.camera.background_color = WHITE
        self.cap = caption(" ")
        self.add(self.cap)

    def say(self, text, run_time=0.5):
        new = caption(text)
        self.play(FadeOut(self.cap, shift=UP * 0.15), FadeIn(new, shift=UP * 0.15), run_time=run_time)
        self.cap = new


def xcorr(X, W, stride=1):
    k = W.shape[0]
    H = (X.shape[0] - k) // stride + 1
    Wd = (X.shape[1] - k) // stride + 1
    return np.array([[(X[i * stride:i * stride + k, j * stride:j * stride + k] * W).sum()
                      for j in range(Wd)] for i in range(H)])


def maxpool(X):
    H, W = X.shape[0] // 2, X.shape[1] // 2
    return np.array([[X[2 * i:2 * i + 2, 2 * j:2 * j + 2].max() for j in range(W)] for i in range(H)])


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) else f"{v:.1f}"


def cell_color(v, vmax):
    if vmax <= 0 or v <= 0:
        return WHITE
    return interpolate_color(ManimColor(WHITE), ManimColor(PURPLE_C1), 0.15 + 0.6 * min(v / vmax, 1))


def grid(values, cell, vmax=None, show=True, font=22, blank=False):
    """Grid of squares with numbers. Returns VGroup of rows; g[i][j] = VGroup(square, text)."""
    values = np.asarray(values)
    vmax = values.max() if vmax is None else vmax
    rows = VGroup()
    for i in range(values.shape[0]):
        row = VGroup()
        for j in range(values.shape[1]):
            v = values[i, j]
            sq = Square(side_length=cell, stroke_color=INK, stroke_width=1.5,
                        fill_color=WHITE if blank else cell_color(v, vmax), fill_opacity=1)
            sq.move_to([j * cell, -i * cell, 0])
            t = Text(fmt(v) if (show and not blank) else "", color=INK, font_size=font).move_to(sq)
            row.add(VGroup(sq, t))
        rows.add(row)
    return rows


def set_cell(g, i, j, v, vmax, font=22):
    sq, t = g[i][j]
    new_t = Text(fmt(v), color=INK, font_size=font).move_to(sq)
    return [sq.animate.set_fill(cell_color(v, vmax), 1), Transform(t, new_t)]


# ---------------------------------------------------------------- 1. 2D convolution sliding
class Conv2DSliding(BaseScene):
    def construct(self):
        self.add(title("2D convolution: one small filter, slid everywhere"))
        X = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0]])
        K1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        K2 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        Z1, Z2 = xcorr(X, K1), xcorr(X, K2)

        c_in, c_out = 0.6, 0.48
        gin = grid(X, c_in, vmax=2.5).move_to([-4.7, -0.15, 0])
        lab_in = Text("image X (6×6)", color=INK, font_size=22).next_to(gin, UP, buff=0.2)
        gk = grid(K1, 0.45, vmax=1.6, font=20).move_to([0.0, 1.35, 0])
        for i in range(3):
            gk[i][2][0].set_fill(ORANGE_C1, 0.55)
        lab_k = Text("filter W (3×3)", color=INK, font_size=22).next_to(gk, UP, buff=0.2)
        gout = grid(np.zeros((4, 4)), c_out, blank=True).move_to([3.6, 1.15, 0])
        lab_out = Text("feature map 1", color=INK, font_size=22).next_to(gout, UP, buff=0.15)

        self.play(FadeIn(gin), FadeIn(lab_in))
        self.say("A 6×6 binary image. The filter has 9 weights (orange = 1, white = 0).")
        self.play(FadeIn(gk), FadeIn(lab_k), FadeIn(gout), FadeIn(lab_out))

        def window_rect(i, j):
            top_left = gin[i][j][0].get_corner(UL)
            r = Rectangle(width=3 * c_in, height=3 * c_in, stroke_color=ORANGE_C1, stroke_width=6)
            r.move_to(top_left + np.array([1.5 * c_in, -1.5 * c_in, 0]))
            return r

        win = window_rect(0, 0)
        eq = MathTex(r"z_{ij}=\sum_{m}\sum_{n} X_{i+m,\,j+n}\,W_{m,n}", color=INK, font_size=28)
        eq.move_to([0.0, -0.55, 0])
        self.play(Create(win), Write(eq))
        readout = Text(" ", font_size=24, color=INK).move_to([0.0, -1.45, 0])
        self.add(readout)
        self.say("Multiply the window by the filter element-wise and sum: an inner product.")

        for idx in range(16):
            i, j = divmod(idx, 4)
            slow = idx < 4
            new_win = window_rect(i, j)
            col = X[i:i + 3, j + 2]
            txt = f"{' + '.join(str(v) for v in col)} = {Z1[i, j]}"
            new_read = Text(txt, font_size=24, color=INK).move_to(readout)
            self.play(Transform(win, new_win), Transform(readout, new_read), run_time=0.45 if slow else 0.18)
            self.play(*set_cell(gout, i, j, Z1[i, j], 3.2, font=20), run_time=0.4 if slow else 0.15)
            if idx == 3:
                self.say("Same 9 weights at every position: that is parameter sharing.")
        self.say("Result: a feature map that lights up where the image has a vertical stroke on the right.")
        self.wait(1.2)

        gout2 = grid(np.zeros((4, 4)), c_out, blank=True).move_to([3.6, -1.75, 0])
        lab_out2 = Text("feature map 2", color=INK, font_size=22).next_to(gout2, UP, buff=0.15)
        gk2 = grid(K2, 0.45, vmax=1.6, font=20).move_to(gk)
        for j in range(3):
            gk2[0][j][0].set_fill(BLUE_C1, 0.45)
        self.say("A second filter (top row = 1) gives a second channel.")
        self.play(FadeOut(win), FadeOut(readout), ReplacementTransform(gk, gk2), FadeIn(gout2), FadeIn(lab_out2))
        anims = []
        for i in range(4):
            for j in range(4):
                anims += set_cell(gout2, i, j, Z2[i, j], 3.2, font=20)
        self.play(*anims, run_time=1.2)
        self.say("C_out filters → C_out channels. The network learns the filter values.")
        self.wait(2)


# ---------------------------------------------------------------- 2. Equivariance and pooling invariance
class EquivariancePooling(BaseScene):
    def construct(self):
        self.add(title("Convolution is equivariant; max pooling adds a little invariance"))
        K = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

        def make_input(r, c):
            X = np.zeros((6, 6))
            X[r, c], X[r + 1, c] = 1, 2
            return X

        cell = 0.36
        xs = [-4.6, -0.6, 2.9]

        def row(X, y, show_pool=False):
            Z = xcorr(X, K)
            P = maxpool(Z)
            gi = grid(X, cell, vmax=2.5, font=16).move_to([xs[0], y, 0])
            go = grid(Z, cell, vmax=3.2, font=16).move_to([xs[1], y, 0])
            gp = grid(P, cell * 1.3, vmax=3.2, font=18).move_to([xs[2] + 0.6, y, 0])
            a1 = Arrow(gi.get_right(), go.get_left(), buff=0.25, color=INK, stroke_width=3)
            a2 = Arrow(go.get_right(), gp.get_left(), buff=0.25, color=INK, stroke_width=3)
            l1 = Text("conv", font_size=18, color=INK).next_to(a1, UP, buff=0.05)
            l2 = Text("2×2 max pool", font_size=18, color=INK).next_to(a2, UP, buff=0.05)
            return VGroup(gi, go, gp, a1, a2, l1, l2), P

        heads = VGroup(*[Text(t, font_size=20, color=GREY_C1).move_to([x if k < 2 else x + 0.6, 2.75, 0])
                         for k, (t, x) in enumerate(zip(["input", "feature map", "pooled"], xs))])
        filt = grid(K, 0.3, vmax=1.6, font=14).to_corner(UR, buff=0.4).shift(DOWN * 0.6)
        flab = Text("filter", font_size=16, color=INK).next_to(filt, DOWN, buff=0.1)

        top, P0 = row(make_input(1, 1), 1.35)
        self.play(FadeIn(heads), FadeIn(filt), FadeIn(flab))
        self.play(FadeIn(top[0]), run_time=0.6)
        self.say("A small pattern in the image.")
        self.play(GrowArrow(top[3]), FadeIn(top[5]), FadeIn(top[1]))
        self.say("Convolving gives a blob in the feature map at the same place.")
        self.wait(0.6)

        bot, P1 = row(make_input(2, 3), -1.45)
        self.say("Shift the input 1 down and 2 right ...")
        self.play(TransformFromCopy(top[0], bot[0]), run_time=1.0)
        self.play(GrowArrow(bot[3]), FadeIn(bot[5]), FadeIn(bot[1]))
        eq = MathTex(r"f(t(x)) = t(f(x))", color=BLUE_C1, font_size=34).move_to([4.6, -0.05, 0])
        self.say("... and the feature map shifts by exactly the same amount: equivariance.")
        self.play(Write(eq))
        self.wait(1.2)

        self.say("Now pool 2×2. Max pooling keeps the strongest response per block.")
        self.play(GrowArrow(top[4]), FadeIn(top[6]), FadeIn(top[2]))
        self.play(GrowArrow(bot[4]), FadeIn(bot[6]), FadeIn(bot[2]))
        self.say("A big shift still moves the pooled output (equivariant, coarser).")
        self.wait(1.0)

        small, P2 = row(make_input(1, 2), -1.45)
        self.say("A 1-pixel shift that stays inside the pooling blocks ...")
        self.play(FadeOut(eq), Transform(bot, small), run_time=1.2)
        same = SurroundingRectangle(VGroup(top[2], bot[2]), color=GREEN_C1, buff=0.15, stroke_width=5)
        eq2 = MathTex(r"f(t(x)) = f(x)", color=GREEN_C1, font_size=34).next_to(same, DOWN, buff=0.1)
        assert np.array_equal(P0, P2)
        self.say("... gives the identical pooled output: local invariance.")
        self.play(Create(same), Write(eq2))
        self.wait(2)


# ---------------------------------------------------------------- 3. Receptive field growth
class ReceptiveField(BaseScene):
    def construct(self):
        self.ttl = title("Receptive field: how much input one unit can see")
        self.add(self.ttl)
        n = 13
        dx = 0.62
        x0 = -dx * (n - 1) / 2 - 1.6
        ys = [-2.3, -0.95, 0.4, 1.75]
        names = ["input x", "layer 1", "layer 2", "layer 3"]

        def pos(layer, k):
            return np.array([x0 + k * dx, ys[layer], 0])

        def build(positions):
            nodes, labels = VGroup(), VGroup()
            for L, ks in enumerate(positions):
                for k in ks:
                    nodes.add(Circle(radius=0.15, stroke_color=GREY_C1, stroke_width=2,
                                     fill_color=WHITE, fill_opacity=1).move_to(pos(L, k)))
                labels.add(Text(names[L], font_size=18, color=INK).next_to(pos(L, 0), LEFT, buff=0.35))
            return nodes, labels

        def cone(top_k, plan):
            """plan: list per layer (from top) of (stride-in-input-units of layer below).
            Returns highlighted node sets per layer and the edges."""
            sets = {3: {top_k}}
            edges = VGroup()
            for L in [3, 2, 1]:
                step = plan[L]
                below = set()
                for k in sets[L]:
                    for o in (-step, 0, step):
                        below.add(k + o)
                        edges.add(Line(pos(L, k), pos(L - 1, k + o), stroke_color=ORANGE_C1, stroke_width=2.5,
                                       buff=0.15))
                sets[L - 1] = below
            return sets, edges

        def hl(sets):
            g = VGroup()
            for L, ks in sets.items():
                for k in ks:
                    g.add(Circle(radius=0.15, stroke_color=ORANGE_C1, stroke_width=3,
                                 fill_color=ORANGE_C1, fill_opacity=0.85).move_to(pos(L, k)))
            return g

        rf_eq = MathTex(r"r_\ell = r_{\ell-1} + (k-1)\,j_{\ell-1}", color=INK, font_size=30)
        j_eq = MathTex(r"j_\ell = j_{\ell-1}\, s_\ell", color=INK, font_size=30)
        eqs = VGroup(rf_eq, j_eq).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.1, 1.6, 0])
        sub = Text("r: receptive field\nj: jump (input px per step)\nk: kernel, s: stride", font_size=16,
                   color=GREY_C1, line_spacing=0.8).next_to(eqs, DOWN, buff=0.25).align_to(eqs, LEFT)

        # Part A: stride 1 everywhere
        all_k = list(range(n))
        nodes, labels = build([all_k] * 4)
        self.play(FadeIn(nodes), FadeIn(labels))
        self.say("Three conv layers, kernel size 3, stride 1 (zero-padded).")
        sets, edges = cone(6, {3: 1, 2: 1, 1: 1})
        rf_lab = Text("", font_size=24, color=ORANGE_C1)
        self.play(FadeIn(eqs), FadeIn(sub))
        top = hl({3: sets[3]})
        self.play(FadeIn(top))
        for L, r in [(2, 3), (1, 5), (0, 7)]:
            lay_edges = VGroup(*[e for e in edges if abs(e.get_end()[1] - ys[L]) < 0.3])
            new_lab = Text(f"sees {r} {'inputs' if L == 0 else 'units below'}" if L else f"receptive field = {r}",
                           font_size=24, color=ORANGE_C1).move_to([5.1, -0.9, 0])
            self.play(Create(lay_edges), FadeIn(hl({L: sets[L]})), Transform(rf_lab, new_lab), run_time=0.9)
            self.add(rf_lab)
        self.say("Each layer adds k − 1 = 2: receptive field 3 → 5 → 7.")
        self.wait(1.5)

        # Part B: stride 2 in layer 2
        keep = {id(self.cap), id(eqs), id(sub), id(self.ttl)}
        self.play(*[FadeOut(m) for m in list(self.mobjects) if id(m) not in keep], run_time=0.6)
        nodes, labels = build([all_k, all_k, list(range(0, n, 2)), list(range(0, n, 2))])
        self.play(FadeIn(nodes), FadeIn(labels))
        self.say("Same depth, but layer 2 uses stride 2 (keeps every other position).")
        sets, edges = cone(6, {3: 2, 2: 1, 1: 1})
        self.play(FadeIn(hl({3: sets[3]})))
        rf_lab = Text("", font_size=24, color=ORANGE_C1)
        for L, txt in [(2, "sees 3 units, 2 px apart"), (1, "sees 7 units below"), (0, "receptive field = 9")]:
            lay_edges = VGroup(*[e for e in edges if abs(e.get_end()[1] - ys[L]) < 0.3])
            new_lab = Text(txt, font_size=24, color=ORANGE_C1).move_to([5.1, -0.9, 0])
            self.play(Create(lay_edges), FadeIn(hl({L: sets[L]})), Transform(rf_lab, new_lab), run_time=0.9)
            self.add(rf_lab)
        self.say("After a stride-2 step every later layer grows the field twice as fast: 3 → 5 → 9.")
        self.wait(2.2)


# ---------------------------------------------------------------- 4. Padding, stride, dilation (1D)
class PaddingStrideDilation(BaseScene):
    def construct(self):
        self.add(title("Padding, stride and dilation (1D, kernel size 3)"))
        x = np.array([1, 3, 2, 0, 4, 1, 2])
        w = np.array([-1, 0, 1])
        n, cell = len(x), 0.7
        xs0 = -cell * (n - 1) / 2 - 1.8
        y_in, y_out = 0.9, -0.9

        def in_pos(k):  # k may be -1 or n for padding
            return np.array([xs0 + k * cell, y_in, 0])

        def box(v, p, color=WHITE, dashed=False):
            sq = Square(side_length=cell * 0.92, stroke_color=GREY_C1 if dashed else INK, stroke_width=1.5,
                        fill_color=color, fill_opacity=1).move_to(p)
            if dashed:
                sq = DashedVMobject(sq, num_dashes=14)
            return VGroup(sq, Text(str(v), font_size=22, color=GREY_C1 if dashed else INK).move_to(p))

        inp = VGroup(*[box(v, in_pos(k)) for k, v in enumerate(x)])
        lab_in = Text("x", font_size=24, color=INK).next_to(inp, LEFT, buff=0.9)
        wlab = MathTex(r"\omega = [-1,\ 0,\ 1]", color=INK, font_size=30).move_to([4.2, 1.9, 0])
        size_eq = MathTex(r"n_{out} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1",
                          color=INK, font_size=28).move_to([4.2, 0.75, 0])
        self.play(FadeIn(inp), FadeIn(lab_in), FadeIn(wlab), FadeIn(size_eq))

        params = Text(" ", font_size=22, color=BLUE_C1).move_to([4.6, -0.3, 0])
        self.add(params)

        def run(label, p, s, d, say_text):
            nonlocal params
            pads = VGroup()
            if p:
                pads = VGroup(box(0, in_pos(-1), dashed=True), box(0, in_pos(n), dashed=True))
                self.play(FadeIn(pads), run_time=0.4)
            xp = np.r_[np.zeros(p), x, np.zeros(p)]
            span = d * (len(w) - 1) + 1
            outs = [(c, float(sum(w[t] * xp[c + t * d] for t in range(3)))) for c in range(0, len(xp) - span + 1, s)]
            n_out = (n + 2 * p - d * 2 - 1) // s + 1
            assert n_out == len(outs)
            new_params = Text(f"{label}\np={p}, s={s}, d={d}  →  n_out = {n_out}", font_size=22,
                              color=BLUE_C1, line_spacing=0.8).move_to([4.2, -0.45, 0])
            self.say(say_text)
            self.play(Transform(params, new_params), run_time=0.4)
            out_boxes = VGroup()
            taps = None
            for idx, (c, v) in enumerate(outs):
                centre = c + d  # index in padded coords of the middle tap
                taps_new = VGroup(*[Circle(radius=cell * 0.42, stroke_color=ORANGE_C1, stroke_width=5)
                                    .move_to(in_pos(c + t * d - p)) for t in range(3)])
                ob = box(fmt(v), np.array([in_pos(centre - p)[0], y_out, 0]), color=cell_color(abs(v), 4.5))
                arrows = VGroup(*[Line(in_pos(c + t * d - p) + DOWN * cell * 0.45, ob.get_top(),
                                       stroke_color=ORANGE_C1, stroke_width=2) for t in range(3)])
                if taps is None:
                    taps = taps_new
                    self.play(Create(taps), run_time=0.3)
                else:
                    self.play(Transform(taps, taps_new), run_time=0.3)
                self.play(Create(arrows), FadeIn(ob), run_time=0.3)
                self.play(FadeOut(arrows), run_time=0.12)
                out_boxes.add(ob)
            self.wait(0.8)
            self.play(FadeOut(taps), FadeOut(out_boxes), FadeOut(pads), run_time=0.5)

        run("zero padding", 1, 1, 1, "Zero-padding: pretend x = 0 outside, so the output keeps the input length.")
        run("valid", 0, 1, 1, "'Valid': only where the kernel fits. Output shrinks by k − 1.")
        run("stride 2", 1, 2, 1, "Stride 2: evaluate every other position, about half the outputs.")
        run("dilation 2", 0, 1, 2, "Dilation 2: holes between the taps. Sees 5 inputs with only 3 weights.")
        self.say("All four use the same 3 weights; only where they are applied changes.")
        self.wait(1.5)
