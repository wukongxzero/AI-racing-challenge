"""Basic Pygame top-down view. Drawing only - it never touches the physics, so the
game can step several cars itself and draw them all here.

    view = TopDownView(env.track)                     # whole track in the window
    view = TopDownView(env.track, zoom=3)             # 3x closer, the camera follows a car
    while view.draw([{"x": .., "y": .., "psi": .., "color": (r, g, b)}], text="lap 1"):
        ...
    view.close()
"""
from __future__ import annotations

import math

import numpy as np

GRASS = (46, 110, 52)
ROAD = (70, 70, 74)
EDGE = (235, 235, 235)
START = (240, 200, 40)


class TopDownView:
    """zoom=1 shows the whole track; zoom > 1 shows a closer view that follows the first car
    (or the point passed as `center`). headless=True draws into an off-screen image (no
    window): use frame() to get it, e.g. for recording videos."""

    def __init__(self, track, window=(1000, 700), margin=1.5, car_length=0.58, car_width=0.31,
                 caption="RacingEnv", headless=False, zoom=1.0):
        import pygame  # imported here so the sim works without pygame installed
        self.pg = pygame
        self.headless = headless
        self.window = window
        pygame.init()
        if headless:
            self.screen = pygame.Surface(window)
        else:
            self.screen = pygame.display.set_mode(window)
            pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("consolas", 18)
        self.car_length, self.car_width = car_length, car_width

        tab = track.centerline_table(ds=0.1, s_min=0.0, s_max=track.length)
        nx, ny = -np.sin(tab["theta"]), np.cos(tab["theta"])
        self._left = np.c_[tab["x"] + nx * tab["w_left"], tab["y"] + ny * tab["w_left"]]
        self._right = np.c_[tab["x"] - nx * tab["w_right"], tab["y"] - ny * tab["w_right"]]

        # world (m) -> screen (px): the whole track fits the window (30 px left for text)
        self.origin = np.minimum(self._left.min(0), self._right.min(0)) - margin
        self._top = np.maximum(self._left.max(0), self._right.max(0)) + margin
        size = self._top - self.origin
        self._fit_scale = min(window[0] / size[0], (window[1] - 30) / size[1])
        self.set_zoom(zoom)

    def set_zoom(self, zoom):
        """1 = whole track; > 1 = closer view following a car. Redraws the track once."""
        pg = self.pg
        self.zoom = float(zoom)
        self.scale = self._fit_scale * self.zoom
        if self.zoom <= 1.0:
            bg_size = self.window                       # whole track = exactly the window
        else:
            size = (self._top - self.origin) * self.scale
            bg_size = (int(size[0]) + 1, int(size[1]) + 1)
        self.height = bg_size[1]
        # pre-render the static track once at this zoom
        self.background = pg.Surface(bg_size)
        self.background.fill(GRASS)
        left, right = self._left, self._right
        for i in range(len(left) - 1):                  # road as quads between the edges
            quad = [self.to_px(*left[i]), self.to_px(*left[i + 1]),
                    self.to_px(*right[i + 1]), self.to_px(*right[i])]
            pg.draw.polygon(self.background, ROAD, quad)
        width = max(2, int(round(2 * self.zoom)))
        pg.draw.lines(self.background, EDGE, True, [self.to_px(*p) for p in left], width)
        pg.draw.lines(self.background, EDGE, True, [self.to_px(*p) for p in right], width)
        pg.draw.line(self.background, START, self.to_px(*left[0]), self.to_px(*right[0]), width + 1)

    def to_px(self, x, y):
        """World point (m) -> pixel on the track picture."""
        return (int((x - self.origin[0]) * self.scale),
                int(self.height - (y - self.origin[1]) * self.scale))

    def draw(self, cars, text="", center=None):
        """Draw cars (dicts with x, y, psi and optional color) and text (several lines with
        '\\n'). When zoomed in, the view is centred on `center` (x, y) or the first car.
        Returns False when the window was closed (or Esc pressed)."""
        pg = self.pg
        if not self.headless:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return False
        # camera: shift the picture so the followed point sits in the middle of the window
        ox = oy = 0
        if self.zoom > 1.0 and (center is not None or cars):
            cx, cy = center if center is not None else (cars[0]["x"], cars[0]["y"])
            px, py = self.to_px(cx, cy)
            ox, oy = px - self.window[0] // 2, py - self.window[1] // 2
            self.screen.fill(GRASS)
        self.screen.blit(self.background, (-ox, -oy))
        for car in cars:
            c, s = math.cos(car["psi"]), math.sin(car["psi"])
            hl, hw = 0.5 * self.car_length, 0.5 * self.car_width
            corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
            pts = [self._shift(self.to_px(car["x"] + c * a - s * b, car["y"] + s * a + c * b), ox, oy)
                   for a, b in corners]
            pg.draw.polygon(self.screen, car.get("color", (220, 40, 40)), pts)
            pg.draw.line(self.screen, (255, 255, 255), self._shift(self.to_px(car["x"], car["y"]), ox, oy),
                         self._shift(self.to_px(car["x"] + c * hl, car["y"] + s * hl), ox, oy),
                         max(2, int(self.zoom)))                                  # nose
        for i, line in enumerate(text.split("\n") if text else []):
            self.screen.blit(self.font.render(line, True, (255, 255, 255), (0, 0, 0)), (10, 8 + 22 * i))
        if not self.headless:
            pg.display.flip()
        return True

    @staticmethod
    def _shift(p, ox, oy):
        return (p[0] - ox, p[1] - oy)

    def frame(self):
        """The last drawn picture as an RGB array, shape (height, width, 3), uint8."""
        return np.ascontiguousarray(self.pg.surfarray.array3d(self.screen).swapaxes(0, 1))

    def close(self):
        if not self.headless:
            self.pg.display.quit()
