"""Manim scenes for Robotics (34753), Week 6 (Trajectory Planning & Manipulator Dynamics).

Render (from this folder):  manim -qm --format mp4 scenes.py <SceneName>
All curves and motions are computed with NumPy from the lecture's formulas, so what you see is what the maths does.
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


def axes(x_range, y_range, w, h, y_step):
    ax = Axes(x_range=x_range, y_range=[*y_range, y_step], x_length=w, y_length=h, tips=False,
              axis_config={"color": INK, "stroke_width": 2, "include_ticks": False},
              y_axis_config={"include_ticks": True, "include_numbers": True, "font_size": 18,
                             "decimal_number_config": {"color": INK, "num_decimal_places": 0}})
    return ax


# ---------------------------------------------------------------- 1. Cubic vs quintic
class CubicVsQuintic(BaseScene):
    def construct(self):
        self.add(title("Cubic vs quintic: what happens to acceleration"))
        T0, T1 = -0.25, 1.25

        def rest(f, before, after):
            return lambda t: before if t < 0 else (after if t > 1 else f(t))

        cub = [rest(lambda t: 3 * t * t - 2 * t ** 3, 0, 1), rest(lambda t: 6 * t - 6 * t * t, 0, 0),
               rest(lambda t: 6 - 12 * t, 0, 0)]
        qui = [rest(lambda t: 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5, 0, 1),
               rest(lambda t: 30 * t ** 2 - 60 * t ** 3 + 30 * t ** 4, 0, 0),
               rest(lambda t: 60 * t - 180 * t ** 2 + 120 * t ** 3, 0, 0)]
        specs = [("position  q", [-0.2, 1.2], 0.5), ("velocity  q̇", [-0.2, 2.0], 1), ("acceleration  q̈", [-7, 7], 5)]
        axs, labels = VGroup(), VGroup()
        for i, (name, yr, st) in enumerate(specs):
            ax = axes([T0, T1, 0.25], yr, 3.7, 3.2, st).move_to([-4.55 + 4.55 * i, 0.15, 0])
            axs.add(ax)
            labels.add(Text(name, color=INK, font_size=24).next_to(ax, UP, buff=0.2))
        self.play(Create(axs), FadeIn(labels))
        zone = VGroup(*[Text("grey band = at rest", color=GREY_C1, font_size=16).next_to(ax, DOWN, buff=0.12)
                        for ax in axs[1:]])
        for ax in axs:
            for x0 in (-0.25, 1.0):
                band = Rectangle(width=abs(ax.c2p(x0 + 0.25, 0)[0] - ax.c2p(x0, 0)[0]), height=3.2,
                                 stroke_width=0, fill_color="#eeeeee", fill_opacity=1)
                band.move_to([(ax.c2p(x0, 0)[0] + ax.c2p(x0 + 0.25, 0)[0]) / 2, ax.get_center()[1], 0])
                self.add(band)
        self.bring_to_front(axs)
        self.add(zone)

        def curves(fs, col):
            g = VGroup()
            for ax, f in zip(axs, fs):
                for a, b in [(T0, -1e-4), (0, 1), (1 + 1e-4, T1)]:
                    g.add(ax.plot(f, x_range=[a, b, 0.005], color=col, stroke_width=4))
            return g

        self.say("Rest-to-rest move from q = 0 to q = 1 in 1 s. First a cubic.")
        cg = curves(cub, BLUE_C1)
        self.play(Create(cg), run_time=2.2)
        eqc = MathTex(r"q=3t^2-2t^3", color=BLUE_C1, font_size=32).next_to(axs[0], DOWN, buff=0.35)
        self.play(Write(eqc))
        self.say("Position and velocity are smooth: 4 conditions, 4 coefficients.")
        self.wait(1.0)
        ax = axs[2]
        j0 = DashedLine(ax.c2p(0, 0), ax.c2p(0, 6), color=RED_C1, stroke_width=4)
        j1 = DashedLine(ax.c2p(1, -6), ax.c2p(1, 0), color=RED_C1, stroke_width=4)
        self.say("But acceleration jumps from 0 to 6 at the start and from −6 to 0 at the end.")
        self.play(Create(j0), Create(j1), Flash(ax.c2p(0, 6), color=RED_C1), Flash(ax.c2p(1, -6), color=RED_C1))
        jl = Text("dashed jumps = infinite jerk", color=RED_C1, font_size=20).next_to(zone[1], DOWN, buff=0.12)
        self.play(FadeIn(jl))
        self.say("Infinite jerk means a torque step: vibration and wear.")
        self.wait(1.2)

        self.say("Quintic: also demand q̈ = 0 at both ends. 6 conditions, 6 coefficients.")
        qg = curves(qui, RED_C1)
        self.play(Create(qg), run_time=2.2)
        eqq = MathTex(r"q=10t^3-15t^4+6t^5", color=RED_C1, font_size=32).next_to(eqc, DOWN, buff=0.15)
        self.play(Write(eqq))
        self.say("Its acceleration starts and ends at 0: no jumps, bounded jerk.")
        self.play(Indicate(qg[6:9], color=RED_C1, scale_factor=1.05))
        self.wait(1.0)
        vax = axs[1]
        pk1 = DashedLine(vax.c2p(0.5, 1.5), vax.c2p(1.25, 1.5), color=BLUE_C1, stroke_width=2)
        pk2 = DashedLine(vax.c2p(0.5, 1.875), vax.c2p(1.25, 1.875), color=RED_C1, stroke_width=2)
        p1 = Text("1.5", color=BLUE_C1, font_size=18).next_to(pk1, RIGHT, buff=0.08)
        p2 = Text("1.875", color=RED_C1, font_size=18).next_to(pk2, RIGHT, buff=0.08)
        self.say("The price: a higher peak velocity (1.875 instead of 1.5).")
        self.play(Create(pk1), Create(pk2), FadeIn(p1), FadeIn(p2))
        self.wait(2)


# ---------------------------------------------------------------- 2. Cubic boundary conditions
class CubicBoundaryConditions(BaseScene):
    def construct(self):
        self.add(title("A cubic segment: 4 conditions fix 4 coefficients"))
        ax = axes([0, 1, 0.25], [-0.5, 1.5], 6.2, 4.6, 0.5).move_to([-3.1, -0.15, 0])
        ax.y_axis.numbers.become(VGroup(*[DecimalNumber(v, num_decimal_places=1, color=INK, font_size=18)
                                          .next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in [-0.5, 0, 0.5, 1, 1.5]]))
        tl = Text("t", color=INK, font_size=24).next_to(ax.x_axis, RIGHT, buff=0.15)
        ql = Text("q", color=INK, font_size=24).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Create(ax), FadeIn(tl), FadeIn(ql))
        qA, qB = 0.0, 1.0
        vA, vB = ValueTracker(0.0), ValueTracker(0.0)

        def coeffs():
            dq = qB - qA
            return qA, vA.get_value(), 3 * dq - 2 * vA.get_value() - vB.get_value(), -2 * dq + vA.get_value() + vB.get_value()

        def curve():
            c0, c1, c2, c3 = coeffs()
            return ax.plot(lambda t: c0 + c1 * t + c2 * t * t + c3 * t ** 3, x_range=[0, 1, 0.005], color=BLUE_C1, stroke_width=5)

        def tangent(t0, q0, v, col):
            p = ax.c2p(t0, q0)
            d = ax.c2p(t0 + 0.1, q0 + 0.1 * v) - p
            d = d / np.linalg.norm(d)
            return Arrow(p - 0.55 * d, p + 0.75 * d, buff=0, color=col, stroke_width=5, max_tip_length_to_length_ratio=0.25)

        A = Dot(ax.c2p(0, qA), radius=0.1, color=INK)
        B = Dot(ax.c2p(1, qB), radius=0.1, color=INK)
        la = MathTex(r"q_A", color=INK, font_size=34).next_to(A, DOWN + RIGHT, buff=0.12)
        lb = MathTex(r"q_B", color=INK, font_size=34).next_to(B, UP, buff=0.2)
        self.say("Start and end positions: 2 conditions.")
        self.play(FadeIn(A, scale=2), FadeIn(B, scale=2), Write(la), Write(lb))
        lin = DashedLine(A.get_center(), B.get_center(), color=GREY_C1, stroke_width=3)
        self.say("A straight line fits them, but the velocity jumps from 0 to 1 at the ends.")
        self.play(Create(lin))
        self.wait(1.0)

        sysm = MathTex(r"\begin{bmatrix}1&0&0&0\\0&1&0&0\\1&1&1&1\\0&1&2&3\end{bmatrix}"
                       r"\begin{bmatrix}c_0\\c_1\\c_2\\c_3\end{bmatrix}="
                       r"\begin{bmatrix}q_A\\v_A\\q_B\\v_B\end{bmatrix}", color=INK, font_size=32)
        sysm.move_to([3.6, 1.35, 0])
        sub = Text("(t_A = 0, t_B = 1)", color=GREY_C1, font_size=18).next_to(sysm, DOWN, buff=0.12)
        cv = always_redraw(curve)
        ta = always_redraw(lambda: tangent(0, qA, vA.get_value(), GREEN_C1))
        tb = always_redraw(lambda: tangent(1, qB, vB.get_value(), RED_C1))
        self.say("Add the end velocities v_A and v_B (arrows): 4 conditions → a cubic.")
        self.play(FadeOut(lin), Create(cv), GrowArrow(ta), GrowArrow(tb), Write(sysm), FadeIn(sub))

        c2l = MathTex(r"c_2 = 3\Delta q-2v_A-v_B =", color=INK, font_size=30)
        c3l = MathTex(r"c_3 = -2\Delta q+v_A+v_B =", color=INK, font_size=30)
        c2l.move_to([3.0, -0.55, 0])
        c3l.next_to(c2l, DOWN, buff=0.3).align_to(c2l, LEFT)
        c2n = always_redraw(lambda: DecimalNumber(coeffs()[2], num_decimal_places=2, color=BLUE_C1, font_size=30)
                            .next_to(c2l, RIGHT, buff=0.15))
        c3n = always_redraw(lambda: DecimalNumber(coeffs()[3], num_decimal_places=2, color=BLUE_C1, font_size=30)
                            .next_to(c3l, RIGHT, buff=0.15))
        vbl = MathTex(r"v_B =", color=RED_C1, font_size=30).next_to(c3l, DOWN, buff=0.45).align_to(c2l, LEFT)
        vbn = always_redraw(lambda: DecimalNumber(vB.get_value(), num_decimal_places=1, color=RED_C1, font_size=30)
                            .next_to(vbl, RIGHT, buff=0.15))
        self.play(Write(c2l), Write(c3l), FadeIn(vbl))
        self.add(c2n, c3n, vbn)
        self.wait(1.0)
        self.say("Solving the system gives c_2 and c_3 directly. Now raise v_B …")
        self.play(vB.animate.set_value(3.0), run_time=1.8)
        self.say("… the cubic must arrive steeply, so it dips first to make room.")
        self.play(vB.animate.set_value(6.0), run_time=1.8)
        low = ax.c2p(0.5, -0.25)
        mk = DashedLine(ax.c2p(0, -0.25), ax.c2p(1, -0.25), color=ORANGE_C1, stroke_width=2)
        ml = Text("dips below q_A", color=ORANGE_C1, font_size=20).next_to(mk, DOWN, buff=0.1)
        self.play(Create(mk), FadeIn(ml), Flash(low, color=ORANGE_C1))
        self.say("Same effect as joint q₂ in the soccer-catcher example: check joint limits.")
        self.wait(1.5)
        self.play(FadeOut(mk), FadeOut(ml))
        self.say("Each joint gets its own cubic; v_B comes from q̇ = J⁺ξ at the catch.")
        self.play(vB.animate.set_value(2.0), vA.animate.set_value(0.0), run_time=1.5)
        self.wait(2)


# ---------------------------------------------------------------- 3. Two-link arm dynamics
M, L, G = 1.0, 1.0, 9.81
LC, IZZ = 0.5, 1.01 / 12


def D_of(q):
    c2 = np.cos(q[1])
    d11 = M * LC ** 2 + M * (L ** 2 + LC ** 2 + 2 * L * LC * c2) + 2 * IZZ
    d12 = M * (LC ** 2 + L * LC * c2) + IZZ
    d22 = M * LC ** 2 + IZZ
    return np.array([[d11, d12], [d12, d22]])


def g_of(q):
    return np.array([M * G * (LC + L) * np.cos(q[0]) + M * G * LC * np.cos(q[0] + q[1]),
                     M * G * LC * np.cos(q[0] + q[1])])


def cq_of(q, qd):
    h = -M * L * LC * np.sin(q[1])
    return np.array([2 * h * qd[0] * qd[1] + h * qd[1] ** 2, -h * qd[0] ** 2])


def energy(q, qd):
    K = 0.5 * qd @ D_of(q) @ qd
    P = M * G * (LC * np.sin(q[0]) + L * np.sin(q[0]) + LC * np.sin(q[0] + q[1]))
    return K, P


def simulate(q0, qd0, tau_fn, T, dt=0.001):
    def f(s):
        q, qd = s[:2], s[2:]
        qdd = np.linalg.solve(D_of(q), tau_fn(q) - cq_of(q, qd) - g_of(q))
        return np.r_[qd, qdd]
    s = np.r_[q0, qd0].astype(float)
    out = [s.copy()]
    for _ in range(int(T / dt)):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(s.copy())
    return np.array(out)[::10]   # 0.01 s per sample


class TwoLinkDynamics(BaseScene):
    def construct(self):
        self.add(title("Two-link arm: D q̈ + C q̇ + g = τ in action"))
        S = 1.55
        base = np.array([-2.6, 0.55, 0])
        ground = Line(base + LEFT * 1.0 + DOWN * 0.25, base + RIGHT * 1.0 + DOWN * 0.25, color=INK)
        tri = Polygon(base, base + DOWN * 0.25 + LEFT * 0.25, base + DOWN * 0.25 + RIGHT * 0.25,
                      color=INK, fill_color="#dddddd", fill_opacity=1, stroke_width=2)
        self.add(ground, tri)

        traj = simulate([0, 0], [0, 0], lambda q: np.zeros(2), 6.0)
        idx = ValueTracker(0)
        state = {"traj": traj}

        def cur():
            tr = state["traj"]
            return tr[min(int(idx.get_value()), len(tr) - 1)]

        def pts(q):
            e = base + S * L * np.array([np.cos(q[0]), np.sin(q[0]), 0])
            tip = e + S * L * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1]), 0])
            return e, tip

        def arm():
            s = cur()
            e, tip = pts(s[:2])
            l1 = Line(base, e, color="#7fc8a9", stroke_width=16)
            l2 = Line(e, tip, color="#7fc8a9", stroke_width=16)
            c1 = Dot((base + e) / 2, radius=0.07, color=BLUE_C1)
            c2 = Dot((e + tip) / 2, radius=0.07, color=BLUE_C1)
            return VGroup(l1, l2, Dot(base, radius=0.09, color=INK), Dot(e, radius=0.09, color=INK), c1, c2)

        eq = MathTex(r"D(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) = \tau", color=INK, font_size=34).move_to([3.7, 2.4, 0])
        taul = MathTex(r"\tau = 0", color=RED_C1, font_size=34).next_to(eq, DOWN, buff=0.25)
        self.play(Write(eq))
        names = ["K (kinetic)", "P (potential)", "E = K + P"]
        cols = [BLUE_C1, GREEN_C1, INK]
        rows = VGroup(*[Text(n, color=c, font_size=22) for n, c in zip(names, cols)])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.0, 0.2, 0])

        def val(k):
            def f():
                s = cur()
                K, P = energy(s[:2], s[2:])
                v = [K, P, K + P][k]
                return DecimalNumber(v, num_decimal_places=1, color=cols[k], font_size=26, include_sign=True) \
                    .next_to(rows[k], RIGHT, buff=0.3).set_x(5.2)
            return f
        unit = Text("J  (m = 1 kg, L = 1 m)", color=GREY_C1, font_size=18).next_to(rows, DOWN, buff=0.3).align_to(rows, LEFT)
        nums = [always_redraw(val(k)) for k in range(3)]
        a = always_redraw(arm)
        self.add(a)
        self.say("Both links horizontal, at rest. Now switch the motors off.")
        self.play(Write(taul), FadeIn(rows), FadeIn(unit))
        self.add(*nums)
        self.wait(0.6)
        tip_trace = TracedPath(lambda: pts(cur()[:2])[1], stroke_color=ORANGE_C1, stroke_width=2, stroke_opacity=0.7)
        self.add(tip_trace)
        self.play(idx.animate.set_value(150), run_time=1.5, rate_func=linear)
        self.say("Gravity alone drives it. Joint 2 is not driven, yet link 2 whips: coupling.")
        self.play(idx.animate.set_value(380), run_time=2.3, rate_func=linear)
        self.say("K and P trade off, E = K + P stays constant: the model conserves energy.")
        self.play(idx.animate.set_value(600), run_time=2.2, rate_func=linear)
        self.wait(0.5)

        # gravity compensation
        self.remove(tip_trace)
        hold = simulate([0, 0], [0, 0], g_of, 3.0)
        state["traj"] = hold
        idx.set_value(0)
        gl = MathTex(r"\tau = g(q)", color=PURPLE_C1, font_size=34).move_to(taul)
        self.say("Back to horizontal. Now let the motors deliver exactly τ = g(q).")
        self.play(ReplacementTransform(taul, gl))
        tv = MathTex(r"\tau_1 = 2mgL = 19.6\ \mathrm{N\,m}", r"\\ \tau_2 = \tfrac12 mgL = 4.9\ \mathrm{N\,m}",
                     color=PURPLE_C1, font_size=28).next_to(unit, DOWN, buff=0.4).align_to(rows, LEFT)
        self.play(Write(tv))
        self.play(idx.animate.set_value(300), run_time=2.0, rate_func=linear)
        self.say("Gravity compensation: the arm floats. g(q) = the static holding torques.")
        self.wait(2)
