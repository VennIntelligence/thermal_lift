"""panel_cluster_v7 scene composer (owner-approved r4 content geometry).

Faithful tcforge port of the round-4 prototype ``scripts/v7_composer_demo.py``
(owner verdict 2026-07-08). It is dispatched from
``geometry.build_scene_mask_with_metadata(scene_composer="panel_cluster_v7")``
and returns a uint8 *draw-canvas* mask (subtractive composition already
resolved: ``add & ~subtract``); the shared tail ``_finalize_scene_mask`` then
inscribes the disc, rotates (order-1) and AA-downsamples to HR coverage — exactly
the v6 path. NO defects are carved here: notches / broken traces / dots / hot
spots / dark blobs all move to the single post-rotation realism stage
(see realism.py + generate_training_pool.py). The composer only emits the
``traces`` / ``panels`` / ``zones`` look-up tables the realism stage consumes.

Geometry authority: the prototype's rasterisation and RNG-draw ORDER are ported
verbatim so ``audit_v7_demo_gates`` G1-G8 hold on the tcforge composer. Windowed
stamps (not full-canvas ``make_rectangle``) are used because dense pad/fin zones
can place hundreds of elements per scene; the windowed rasteriser matches the
prototype pixel-for-pixel at the same SSAA and keeps per-scene cost bounded.
The only intended difference from the prototype is SSAA (config ssaa_factor=4 vs
demo SSAA=3) — a strictly finer anti-alias, band-honest and gate-neutral.

FLOOR (28um) and pitch_floor (32um) are derived from the detector pitch
(x1.4 / x1.6, identical to geometry.py:993-995) — no new constants, no red-line
change.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PANEL_ROLES = ("lined", "frame", "windowed", "textured")

_SEC_FLOOR = 34.0  # order-0 rotation staircase margin for the rotated secondary part


class _PanelClusterComposer:
    """Stateful port of the prototype composer (one instance == one scene)."""

    def __init__(self, rng, params, draw_shape, canvas_w_um, canvas_h_um,
                 detector_pitch_um):
        self.rng = rng
        self.params = dict(params or {})
        self.draw_shape = (int(draw_shape[0]), int(draw_shape[1]))
        self.W_UM = float(canvas_w_um)
        self.H_UM = float(canvas_h_um)
        self.DISC_R_UM = min(self.W_UM, self.H_UM) / 2.0
        # um per draw pixel — square pixels, derived from the canvas (== hr_pitch/aa).
        self.UPD = float(canvas_w_um) / float(self.draw_shape[1])
        det = float(detector_pitch_um)
        self.FLOOR = det * 1.4          # 28um universal finest-feature floor
        self.PITCH_FLOOR = det * 1.6    # 32um pad/array pitch floor

        self.add = np.zeros(self.draw_shape, dtype=np.uint8)
        self.subtract = np.zeros(self.draw_shape, dtype=np.uint8)
        self.panels: list[dict] = []
        self.traces: list[dict] = []
        self.zones: list[dict] = []
        self.secondary = False

        p = self.params
        self.secondary_part_p = float(p.get("secondary_part_p", 0.25))
        self.merge_p = float(p.get("merge_p", 0.25))
        sg = p.get("street_gap_um", [40.0, 160.0])
        self.street_lo, self.street_hi = float(sg[0]), float(sg[1])
        self.bridge_p = float(p.get("bridge_p", 0.55))
        rw = p.get("role_weights",
                   {"lined": 0.30, "frame": 0.20, "windowed": 0.25, "textured": 0.25})
        role_p = np.array([float(rw.get(r, 0.0)) for r in PANEL_ROLES], dtype=np.float64)
        self.role_p = (role_p / role_p.sum()) if role_p.sum() > 0 else np.full(4, 0.25)
        self.texture_max_frac = float(p.get("texture_panel_max_frac", 0.334))
        self.tier_weights = dict(p.get("tier_weights",
                                       {"mid": 0.45, "high": 0.45, "xl": 0.10}))
        self.force_tier = p.get("force_tier")

    # ── low-level windowed stamps (um coords, cy=row, cx=col) ─────────────────
    def _window(self, cy, cx, h, w):
        upd, ds = self.UPD, self.draw_shape
        y0 = max(int((cy - h / 2) / upd) - 1, 0)
        y1 = min(int((cy + h / 2) / upd) + 2, ds[0])
        x0 = max(int((cx - w / 2) / upd) - 1, 0)
        x1 = min(int((cx + w / 2) / upd) + 2, ds[1])
        if y1 <= y0 or x1 <= x0:
            return None
        yy = (np.arange(y0, y1, dtype=np.float32) + 0.5) * upd - cy
        xx = (np.arange(x0, x1, dtype=np.float32) + 0.5) * upd - cx
        return (slice(y0, y1), slice(x0, x1)), yy[:, None], xx[None, :]

    def _stamp_rect(self, target, cy, cx, h, w):
        r = self._window(cy, cx, h, w)
        if r is None:
            return
        win, yy, xx = r
        target[win] |= ((np.abs(yy) <= h / 2) & (np.abs(xx) <= w / 2)).astype(np.uint8)

    def _stamp_disc(self, target, cy, cx, d):
        r = self._window(cy, cx, d, d)
        if r is None:
            return
        win, yy, xx = r
        target[win] |= ((yy ** 2 + xx ** 2) <= (d / 2) ** 2).astype(np.uint8)

    def rect(self, cy, cx, h, w, mode="add"):
        self._stamp_rect(self.add if mode == "add" else self.subtract, cy, cx, h, w)

    def disc(self, cy, cx, d, mode="add"):
        self._stamp_disc(self.add if mode == "add" else self.subtract, cy, cx, d)

    def well_with_elements(self, cy, cx, zh, zw, elements):
        well = np.zeros(self.draw_shape, dtype=np.uint8)
        self._stamp_rect(well, cy, cx, zh, zw)
        self.subtract |= well & (1 - elements)
        self.add |= elements

    # ── rng helpers (mirror demo _u/_di) ──────────────────────────────────────
    def _u(self, lo, hi):
        return float(self.rng.uniform(lo, hi))

    def _di(self, lo, hi):
        return int(self.rng.integers(lo, hi + 1))

    # ── zone / texture patches ────────────────────────────────────────────────
    def _pad_zone(self, cy, cx, zh, zw, *, on_die, w_floor):
        rng = self.rng
        FLOOR = self.FLOOR
        pad = self._u(w_floor, 46.0)
        pitch = pad + self._u(FLOOR, 46.0)
        jit = pitch * self._u(0.02, 0.07)
        drop = self._u(0.04, 0.16)
        shape_round = rng.random() < 0.5
        nr = max(int((zh - pad) // pitch), 2)
        nc = max(int((zw - pad) // pitch), 2)
        y0 = cy - (nr - 1) * pitch / 2
        x0 = cx - (nc - 1) * pitch / 2
        elements = np.zeros(self.draw_shape, dtype=np.uint8) if on_die else None
        target = elements if on_die else self.add
        for i in range(nr):
            for j in range(nc):
                if rng.random() < drop:
                    continue
                py = y0 + i * pitch + rng.normal(0.0, jit)
                px = x0 + j * pitch + rng.normal(0.0, jit)
                if shape_round:
                    self._stamp_disc(target, py, px, pad)
                else:
                    self._stamp_rect(target, py, px, pad, pad)
        if on_die:
            self.well_with_elements(cy, cx, zh, zw, elements)

    def _comb_patch(self, cy, cx, zh, zw, horiz, w_floor):
        rng = self.rng
        FLOOR = self.FLOOR
        cw = self._u(w_floor, 44.0) if rng.random() < 0.7 else self._u(44.0, 80.0)
        pitch = cw + self._u(FLOOR + 6, 96.0)
        span, run = (zh, zw) if horiz else (zw, zh)
        n = max(int(span // pitch), 2)
        offs = (np.arange(n) - (n - 1) / 2) * pitch
        for i, off in enumerate(offs):
            frac = self._u(0.6, 0.95)
            side = -1.0 if i % 2 == 0 else 1.0
            if horiz:
                self.rect(cy + off, cx + side * (run - run * frac) / 2, cw, run * frac, mode="sub")
            else:
                self.rect(cy + side * (run - run * frac) / 2, cx + off, run * frac, cw, mode="sub")

    def _fin_zone(self, cy, cx, zh, zw, w_floor):
        rng = self.rng
        FLOOR = self.FLOOR
        fw = self._u(w_floor, 46.0)
        pitch = fw + self._u(FLOOR, 52.0)
        vertical = rng.random() < 0.5
        span = zw if vertical else zh
        n = max(int((span - fw) // pitch), 2)
        offs = (np.arange(n) - (n - 1) / 2) * pitch
        elements = np.zeros(self.draw_shape, dtype=np.uint8)
        for off in offs:
            if rng.random() < 0.06:
                continue
            o = off + rng.normal(0.0, pitch * 0.03)
            ln = (zh if vertical else zw) * self._u(0.82, 0.94)
            if vertical:
                self._stamp_rect(elements, cy, cx + o, ln, fw)
            else:
                self._stamp_rect(elements, cy + o, cx, fw, ln)
        self.well_with_elements(cy, cx, zh, zw, elements)

    def _edge_pad_row(self, panel, edge):
        rng = self.rng
        FLOOR = self.FLOOR
        dy, dx, dh, dw = panel
        pad = self._u(34.0, 56.0)
        pitch = pad + self._u(FLOOR, 44.0)
        strip = pad + 2 * FLOOR
        inset = strip / 2 + FLOOR
        elements = np.zeros(self.draw_shape, dtype=np.uint8)
        if edge in (0, 1):
            cy = dy + (dh / 2 - inset) * (1 if edge else -1)
            run = dw * 0.86
            n = max(int(run // pitch), 3)
            for j in range(n):
                if rng.random() < 0.05:
                    continue
                px = dx - (n - 1) * pitch / 2 + j * pitch + rng.normal(0, pitch * 0.03)
                self._stamp_rect(elements, cy, px, pad, pad)
            self.well_with_elements(cy, dx, strip, run, elements)
            self.zones.append(_zone(cy, dx, strip, run, "edge_pads"))
        else:
            cx = dx + (dw / 2 - inset) * (1 if edge == 3 else -1)
            run = dh * 0.86
            n = max(int(run // pitch), 3)
            for j in range(n):
                if rng.random() < 0.05:
                    continue
                py = dy - (n - 1) * pitch / 2 + j * pitch + rng.normal(0, pitch * 0.03)
                self._stamp_rect(elements, py, cx, pad, pad)
            self.well_with_elements(dy, cx, run, strip, elements)
            self.zones.append(_zone(dy, cx, run, strip, "edge_pads"))

    @staticmethod
    def _boxes_clear(y, x, bh, bw, boxes, frac=0.25):
        for (oy, ox, oh, ow) in boxes:
            iy = max(0.0, min(y + bh / 2, oy + oh / 2) - max(y - bh / 2, oy - oh / 2))
            ix = max(0.0, min(x + bw / 2, ox + ow / 2) - max(x - bw / 2, ox - ow / 2))
            if iy * ix > frac * min(bh * bw, oh * ow):
                return False
        return True

    def _trace_w(self, w_floor):
        return self._u(w_floor, 60.0) if self.rng.random() < 0.6 else self._u(60.0, 150.0)

    def _void_with_traces(self, vy, vx, vh, vw, overhang, island, w_floor):
        rng = self.rng
        elements = np.zeros(self.draw_shape, dtype=np.uint8)
        if island:
            ih = vh * self._u(0.28, 0.50)
            iw = vw * self._u(0.28, 0.50)
            self._stamp_rect(elements, vy, vx, ih, iw)
        for _ in range(self._di(1, 3)):
            tw_ = self._trace_w(w_floor)
            if rng.random() < 0.5:                     # horizontal crossing
                ty = vy + vh * self._u(-0.35, 0.35)
                self._stamp_rect(elements, ty, vx, tw_, vw + 2 * overhang)
                self.traces.append(_trace(ty, vx, tw_, vw, "void_span"))
            else:                                      # vertical crossing
                tx = vx + vw * self._u(-0.35, 0.35)
                self._stamp_rect(elements, vy, tx, vh + 2 * overhang, tw_)
                self.traces.append(_trace(vy, tx, vh, tw_, "void_span"))
        self.well_with_elements(vy, vx, vh, vw, elements)

    def _lined_panel(self, y, x, h, w, w_floor):
        rng = self.rng
        FLOOR = self.FLOOR
        n_lines = int(np.clip(round(h * w / 2.0e6 * self._u(2.0, 4.0)), 3, 14))
        placed_h: list[tuple[float, float]] = []
        placed_v: list[tuple[float, float]] = []
        for _ in range(n_lines):
            sw = self._u(w_floor, 90.0)
            horiz = rng.random() < 0.5
            placed = placed_h if horiz else placed_v
            span = (h if horiz else w)
            for _try in range(8):
                off = span * self._u(-0.38, 0.38)
                if all(abs(off - o) >= hw + sw / 2 + FLOOR for o, hw in placed):
                    break
            else:
                continue
            placed.append((off, sw / 2))
            if horiz:
                self.rect(y + off, x + w * self._u(-0.20, 0.20),
                          sw, w * self._u(0.35, 0.90), mode="sub")
            else:
                self.rect(y + h * self._u(-0.20, 0.20), x + off,
                          h * self._u(0.35, 0.90), sw, mode="sub")
        if min(h, w) > 1600.0 and rng.random() < 0.6:
            wh, ww = h * self._u(0.18, 0.30), w * self._u(0.18, 0.30)
            self.rect(y + h * self._u(-0.25, 0.25), x + w * self._u(-0.25, 0.25),
                      wh, ww, mode="sub")

    def _apply_panel_role(self, y, x, h, w, role, w_floor):
        rng = self.rng
        FLOOR = self.FLOOR
        self.rect(y, x, h, w, mode="add")
        small = min(h, w) < 900.0
        if role == "frame":
            b = min(self._u(50.0, 130.0), min(h, w) * 0.25)
            iv_h, iv_w = h - 2 * b, w - 2 * b
            if iv_h > 3 * FLOOR and iv_w > 3 * FLOOR:
                island = rng.random() < 0.45 and min(iv_h, iv_w) > 6 * FLOOR
                self._void_with_traces(y, x, iv_h, iv_w, b, island, w_floor=w_floor)
        elif role == "windowed":
            placed_w: list[tuple[float, float, float, float]] = []
            for _ in range(self._di(1, 2)):
                wh = h * self._u(0.30, 0.52)
                ww = w * self._u(0.30, 0.52)
                for _try in range(8):
                    wy = y + (h - wh) * self._u(-0.35, 0.35)
                    wx = x + (w - ww) * self._u(-0.35, 0.35)
                    if all(abs(wy - oy) >= (wh + oh) / 2 + FLOOR
                           or abs(wx - ox) >= (ww + ow) / 2 + FLOOR
                           for oy, ox, oh, ow in placed_w):
                        break
                else:
                    continue
                placed_w.append((wy, wx, wh, ww))
                self._void_with_traces(wy, wx, wh, ww, 40.0,
                                       island=bool(rng.random() < 0.15
                                                   and min(wh, ww) > 6 * FLOOR),
                                       w_floor=w_floor)
            if rng.random() < 0.5 and not small:
                self._lined_panel(y, x, h, w, w_floor=w_floor)
        elif role == "textured":
            zh, zw = h * self._u(0.70, 0.85), w * self._u(0.70, 0.85)
            kind = rng.choice(["comb", "pads", "fins"], p=[0.5, 0.3, 0.2])
            if kind == "comb":
                self._comb_patch(y, x, zh, zw, horiz=bool(rng.random() < 0.5), w_floor=w_floor)
            elif kind == "pads":
                self._pad_zone(y, x, zh, zw, on_die=True, w_floor=w_floor)
                self.zones.append(_zone(y, x, zh, zw, "pads"))
            else:
                self._fin_zone(y, x, zh, zw, w_floor=w_floor)
                self.zones.append(_zone(y, x, zh, zw, "fins"))
        else:  # lined
            if not small or rng.random() < 0.5:
                self._lined_panel(y, x, h, w, w_floor=w_floor)

    def _cluster_scene(self, *, tier, force_xl=False):
        rng = self.rng
        H_UM, W_UM, DISC_R_UM = self.H_UM, self.W_UM, self.DISC_R_UM
        if force_xl:
            K, s_lo, s_hi, g_hi = self._di(10, 14), 0.22, 0.40, 50.0
        elif tier == "high":
            K, s_lo, s_hi, g_hi = self._di(6, 9), 0.14, 0.34, self.street_hi
        else:
            K, s_lo, s_hi, g_hi = self._di(3, 5), 0.08, 0.20, self.street_hi

        shrink = 1.0

        def _size():
            return (H_UM * self._u(s_lo, s_hi) * shrink,
                    W_UM * self._u(s_lo, s_hi) * shrink)

        panels: list[tuple] = []
        adjac: list[tuple] = []
        if force_xl:
            h0, w0 = H_UM * self._u(0.60, 0.70), W_UM * self._u(0.60, 0.70)
        else:
            h0, w0 = _size()
        panels.append((H_UM / 2 + H_UM * self._u(-0.06, 0.06),
                       W_UM / 2 + W_UM * self._u(-0.06, 0.06), h0, w0))
        fails = 0
        while len(panels) < K and fails < 8:
            placed = False
            for _try in range(30):
                i = int(rng.integers(0, len(panels)))
                py, px, ph, pw = panels[i]
                hh, ww = _size()
                merge_p = 0.50 if force_xl else self.merge_p
                gap = (-10.0 if rng.random() < merge_p else self._u(self.street_lo, g_hi))
                side = int(rng.integers(0, 4))
                if side in (0, 1):
                    off_max = (pw + ww) / 2 - max(0.3 * min(pw, ww), 200.0)
                    ny = py + ((ph + hh) / 2 + gap) * (-1 if side == 0 else 1)
                    nx = px + self._u(-off_max, off_max)
                else:
                    off_max = (ph + hh) / 2 - max(0.3 * min(ph, hh), 200.0)
                    nx = px + ((pw + ww) / 2 + gap) * (-1 if side == 2 else 1)
                    ny = py + self._u(-off_max, off_max)
                if np.hypot(ny - H_UM / 2, nx - W_UM / 2) + 0.7 * np.hypot(hh, ww) / 2 \
                        > DISC_R_UM * 0.95:
                    continue
                others = ([p for k2, p in enumerate(panels) if k2 != i]
                          if gap < 0 else panels)
                if not self._boxes_clear(ny, nx, hh, ww, others, frac=0.02):
                    continue
                panels.append((ny, nx, hh, ww))
                adjac.append((i, len(panels) - 1, gap, side))
                placed = True
                break
            if placed:
                fails = 0
                shrink = 1.0
            else:
                fails += 1
                shrink = max(shrink * 0.85, 0.45)

        cy_m = float(np.mean([p[0] for p in panels]))
        cx_m = float(np.mean([p[1] for p in panels]))
        sy, sx = H_UM / 2 - cy_m, W_UM / 2 - cx_m
        for scale in (1.0, 0.7, 0.4, 0.2, 0.0):
            ok = all(np.hypot(p[0] + sy * scale - H_UM / 2,
                              p[1] + sx * scale - W_UM / 2)
                     + 0.7 * np.hypot(p[2], p[3]) / 2
                     <= DISC_R_UM * 0.97 for p in panels)
            if ok:
                panels = [(p[0] + sy * scale, p[1] + sx * scale, p[2], p[3])
                          for p in panels]
                break

        roles = [str(rng.choice(PANEL_ROLES, p=self.role_p)) for _ in panels]
        if not any(r in ("frame", "windowed") for r in roles):
            roles[int(rng.integers(0, len(roles)))] = "windowed"
        if "textured" not in roles:
            lined = [k for k, r in enumerate(roles) if r == "lined"]
            pick = (lined[int(rng.integers(0, len(lined)))] if lined
                    else int(rng.integers(0, len(roles))))
            roles[pick] = "textured"
        tex_idx = [k for k, r in enumerate(roles) if r == "textured"]
        # Truncation (not round) to match the prototype's len(panels)//3 exactly for
        # every panel count 3-14 at the default 0.334; the knob stays honoured.
        cap = max(1, int(len(panels) * self.texture_max_frac))
        for k in tex_idx[cap:]:
            roles[k] = "lined"
        if force_xl:
            roles = [(r if r == "textured" else "lined") for r in roles]
        for (py, px, ph, pw), role in zip(panels, roles):
            self._apply_panel_role(py, px, ph, pw, role, w_floor=self.FLOOR)
            self.panels.append(_panel(py, px, ph, pw, role))

        for (i, j, gap, side) in adjac:
            if gap <= 0 or rng.random() > self.bridge_p:
                continue
            yi, xi, hi_, wi = panels[i]
            yj, xj, hj, wj = panels[j]
            span = gap + 60.0
            if side in (0, 1):
                lo = max(xi - wi / 2, xj - wj / 2)
                hi = min(xi + wi / 2, xj + wj / 2)
                if hi - lo < 120.0:
                    continue
                cy_s = (yi + (hi_ / 2) * (-1 if side == 0 else 1)
                        + yj + (hj / 2) * (1 if side == 0 else -1)) / 2
                if rng.random() < 0.6:
                    bw_ = self._u(60.0, 180.0)
                    bx = self._u(lo + bw_ / 2, hi - bw_ / 2)
                    self.rect(cy_s, bx, span, bw_, mode="add")
                    self.traces.append(_trace(cy_s, bx, span, bw_, "bridge"))
                else:
                    n_b = self._di(2, 4)
                    tw_ = self._u(self.FLOOR, 50.0)
                    for b in range(n_b):
                        bx = lo + (b + 1) * (hi - lo) / (n_b + 1)
                        self.rect(cy_s, bx, span, tw_, mode="add")
                        self.traces.append(_trace(cy_s, bx, span, tw_, "bus"))
            else:
                lo = max(yi - hi_ / 2, yj - hj / 2)
                hi = min(yi + hi_ / 2, yj + hj / 2)
                if hi - lo < 120.0:
                    continue
                cx_s = (xi + (wi / 2) * (-1 if side == 2 else 1)
                        + xj + (wj / 2) * (1 if side == 2 else -1)) / 2
                if rng.random() < 0.6:
                    bw_ = self._u(60.0, 180.0)
                    by = self._u(lo + bw_ / 2, hi - bw_ / 2)
                    self.rect(by, cx_s, bw_, span, mode="add")
                    self.traces.append(_trace(by, cx_s, bw_, span, "bridge"))
                else:
                    n_b = self._di(2, 4)
                    tw_ = self._u(self.FLOOR, 50.0)
                    for b in range(n_b):
                        by = lo + (b + 1) * (hi - lo) / (n_b + 1)
                        self.rect(by, cx_s, tw_, span, mode="add")
                        self.traces.append(_trace(by, cx_s, tw_, span, "bus"))

        for _ in range(self._di(0, 3)):
            if len(panels) < 2:
                break
            i, j = rng.choice(len(panels), size=2, replace=False)
            yi, xi, *_r1 = panels[int(i)]
            yj, xj, *_r2 = panels[int(j)]
            if (abs(yi - yj) + abs(xi - xj)) < 300.0:
                continue
            tw_ = self._u(self.FLOOR, 60.0) if rng.random() < 0.6 else self._u(60.0, 150.0)
            self.rect(yi, (xi + xj) / 2, tw_, abs(xj - xi) + tw_, mode="add")
            self.rect((yi + yj) / 2, xj, abs(yj - yi) + tw_, tw_, mode="add")
            self.traces.append(_trace(yi, (xi + xj) / 2, tw_, abs(xj - xi) + tw_, "l_trace"))
            self.traces.append(_trace((yi + yj) / 2, xj, abs(yj - yi) + tw_, tw_, "l_trace"))

        if tier == "high" and rng.random() < 0.4:
            big_p = max(panels, key=lambda p: p[2] * p[3])
            self._edge_pad_row(big_p, edge=int(rng.integers(0, 4)))

    def _secondary_part(self):
        rng = self.rng
        H_UM, W_UM, DISC_R_UM, UPD = self.H_UM, self.W_UM, self.DISC_R_UM, self.UPD
        part = _PanelClusterComposer(self.rng, self.params, self.draw_shape,
                                     self.W_UM, self.H_UM, self.FLOOR / 1.4)
        fr = self._u(0.10, 0.16)
        ph, pw = H_UM * fr, W_UM * fr * self._u(0.8, 1.2)
        ref = [(p["cy_um"], p["cx_um"]) for p in self.panels] or [(H_UM / 2, W_UM / 2)]
        ref_y = float(np.mean([b[0] for b in ref]))
        ref_x = float(np.mean([b[1] for b in ref]))
        phi = np.arctan2(H_UM / 2 - ref_y, W_UM / 2 - ref_x) + self._u(-0.5, 0.5)
        half_diag = np.hypot(pw * 2.2, ph) / 2
        rad = min(DISC_R_UM * self._u(0.55, 0.72), DISC_R_UM * 0.94 - half_diag)
        py = H_UM / 2 + rad * np.sin(phi)
        px = W_UM / 2 + rad * np.cos(phi)

        gap = self._u(40.0, 120.0)
        ph2, pw2 = ph * self._u(0.6, 0.9), pw * self._u(0.5, 0.8)
        x2 = px + (pw + pw2) / 2 + gap
        part._apply_panel_role(py, px, ph, pw,
                               str(rng.choice(["lined", "windowed", "textured"])),
                               w_floor=_SEC_FLOOR)
        part._apply_panel_role(py, x2, ph2, pw2,
                               str(rng.choice(["lined", "frame"])), w_floor=_SEC_FLOOR)
        bw_ = self._u(50.0, 140.0)
        part.rect(py, (px + x2) / 2, bw_, abs(x2 - px) + bw_, mode="add")
        part.traces.append(_trace(py, (px + x2) / 2, bw_, abs(x2 - px) + bw_, "bus"))

        ang = self._u(18.0, 72.0) * (1 if rng.random() < 0.5 else -1)
        half = int((half_diag + 600.0) / UPD)
        cy_px, cx_px = int(py / UPD), int((px + x2) / 2 / UPD)
        y0, y1 = max(cy_px - half, 0), min(cy_px + half, self.draw_shape[0])
        x0, x1 = max(cx_px - half, 0), min(cx_px + half, self.draw_shape[1])
        for src, dst in ((part.add, self.add), (part.subtract, self.subtract)):
            wnd = src[y0:y1, x0:x1]
            rot = ndimage.rotate(wnd, ang, reshape=False, order=0, mode="constant")
            dst[y0:y1, x0:x1] |= rot
        # Record the secondary traces at their post-window-rotation um centres,
        # carrying angle_deg=ang so the realism trace-break stage can resolve the
        # compound (part+scene) orientation. Zones are intentionally NOT recorded
        # (rotated-zone rasterisation complexity; documented in the plan §2.2).
        win_cy_um = (y0 + y1 - 1) / 2.0 * UPD
        win_cx_um = (x0 + x1 - 1) / 2.0 * UPD
        for t in part.traces:
            oy, ox = _rotate_point_forward(t["cy_um"], t["cx_um"],
                                           win_cy_um, win_cx_um, ang)
            self.traces.append(_trace(oy, ox, t["h_um"], t["w_um"], t["kind"], angle_deg=ang))
        self.secondary = True

    def _clutter(self):
        rng = self.rng
        H_UM, W_UM, FLOOR = self.H_UM, self.W_UM, self.FLOOR
        for _ in range(self._di(0, 3)):
            self.disc(H_UM * self._u(0.15, 0.85), W_UM * self._u(0.12, 0.88),
                      self._u(34.0, 80.0), mode="add")
        for _ in range(self._di(0, 2)):
            y, x = H_UM * self._u(0.15, 0.85), W_UM * self._u(0.12, 0.88)
            tl = self._u(60.0, 120.0)
            twd = self._u(FLOOR, 70.0)
            gap = twd + self._u(FLOOR, 60.0)
            if rng.random() < 0.5:
                self.rect(y, x - gap / 2, tl, twd, mode="add")
                self.rect(y, x + gap / 2, tl, twd, mode="add")
            else:
                self.rect(y - gap / 2, x, twd, tl, mode="add")
                self.rect(y + gap / 2, x, twd, tl, mode="add")

    def compose(self):
        rng = self.rng
        if self.force_tier is not None:
            tier = str(self.force_tier)
        else:
            tiers = list(self.tier_weights.keys())
            p = np.array([float(self.tier_weights[t]) for t in tiers], dtype=np.float64)
            p = p / p.sum()
            tier = str(rng.choice(tiers, p=p))
        force_xl = (tier == "xl")
        want_secondary = (not force_xl) and (rng.random() < self.secondary_part_p)
        self._cluster_scene(tier=("high" if force_xl else tier), force_xl=force_xl)
        if want_secondary:
            self._secondary_part()
        self._clutter()
        mask = (self.add & ~self.subtract).astype(np.uint8)
        extra = {
            "scene_tier": tier,
            "panels": self.panels,
            "traces": self.traces,
            "zones": self.zones,
            "secondary_part": bool(self.secondary),
        }
        primitives = [dict(type="v7_panel", **pan) for pan in self.panels]
        return mask, primitives, extra


def _panel(cy, cx, h, w, role):
    return {"cy_um": float(cy), "cx_um": float(cx), "h_um": float(h),
            "w_um": float(w), "role": str(role)}


def _trace(cy, cx, h, w, kind, angle_deg=0.0):
    return {"cy_um": float(cy), "cx_um": float(cx), "h_um": float(h),
            "w_um": float(w), "angle_deg": float(angle_deg), "kind": str(kind)}


def _zone(cy, cx, h, w, kind):
    return {"cy_um": float(cy), "cx_um": float(cx), "h_um": float(h),
            "w_um": float(w), "kind": str(kind)}


def _rotate_point_forward(y, x, cy, cx, angle_deg):
    """Forward map (input->output) of a point under ndimage.rotate(reshape=False).

    Kept consistent with realism._ndimage_rotate_point_forward (calibrated by
    test_carve_trace_breaks_rotation_mapping). Duplicated tiny helper avoids a
    composer->realism import; both share the SAME convention."""
    t = np.radians(float(angle_deg))
    c, s = float(np.cos(t)), float(np.sin(t))
    dy, dx = float(y) - float(cy), float(x) - float(cx)
    return float(cy) + c * dy - s * dx, float(cx) + s * dy + c * dx


def compose_panel_cluster_scene(rng, *, params, common, draw_shape,
                                canvas_w_um, canvas_h_um, detector_pitch_um):
    """Compose one panel_cluster_v7 scene.

    Returns (mask_uint8_draw_canvas, primitives_list, extra_meta) where
    extra_meta = {scene_tier, panels, traces, zones, secondary_part}. ``common``
    is accepted for signature parity with the v6 composer but not needed by the
    windowed rasteriser (draw geometry is fully determined by draw_shape +
    canvas dims); it is intentionally unused."""
    del common  # windowed stamps derive draw geometry from draw_shape/canvas dims
    composer = _PanelClusterComposer(
        rng, params, draw_shape, canvas_w_um, canvas_h_um, detector_pitch_um)
    return composer.compose()
