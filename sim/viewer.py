"""Basic Pygame top-down view. Drawing only - it never touches the physics, so the
game can step several cars itself and draw them all here.

    view = TopDownView(env.track)
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
    def __init__(self, track, window=(1000, 700), margin=1.5, car_length=0.58, car_width=0.31,
                 caption="RacingEnv"):
        import pygame  # imported here so the sim works without pygame installed
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window)
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("consolas", 18)
        self.car_length, self.car_width = car_length, car_width

        tab = track.centerline_table(ds=0.1, s_min=0.0, s_max=track.length)
        nx, ny = -np.sin(tab["theta"]), np.cos(tab["theta"])
        left = np.c_[tab["x"] + nx * tab["w_left"], tab["y"] + ny * tab["w_left"]]
        right = np.c_[tab["x"] - nx * tab["w_right"], tab["y"] - ny * tab["w_right"]]

        # world (m) -> screen (px): fit the track with a margin, y axis up
        lo = np.minimum(left.min(0), right.min(0)) - margin
        hi = np.maximum(left.max(0), right.max(0)) + margin
        self.scale = min((window[0]) / (hi[0] - lo[0]), (window[1] - 30) / (hi[1] - lo[1]))
        self.origin = lo
        self.height = window[1]

        # pre-render the static track once
        self.background = pygame.Surface(window)
        self.background.fill(GRASS)
        for i in range(len(tab["s"]) - 1):  # road as quads between the edges
            quad = [self.to_px(*left[i]), self.to_px(*left[i + 1]),
                    self.to_px(*right[i + 1]), self.to_px(*right[i])]
            pygame.draw.polygon(self.background, ROAD, quad)
        pygame.draw.lines(self.background, EDGE, True, [self.to_px(*p) for p in left], 2)
        pygame.draw.lines(self.background, EDGE, True, [self.to_px(*p) for p in right], 2)
        pygame.draw.line(self.background, START, self.to_px(*left[0]), self.to_px(*right[0]), 3)

    def to_px(self, x, y):
        return (int((x - self.origin[0]) * self.scale),
                int(self.height - (y - self.origin[1]) * self.scale))

    def draw(self, cars, text=""):
        """Draw cars (dicts with x, y, psi and optional color). Returns False when
        the window was closed (or Esc pressed)."""
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                return False
        self.screen.blit(self.background, (0, 0))
        for car in cars:
            c, s = math.cos(car["psi"]), math.sin(car["psi"])
            hl, hw = 0.5 * self.car_length, 0.5 * self.car_width
            corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
            pts = [self.to_px(car["x"] + c * a - s * b, car["y"] + s * a + c * b) for a, b in corners]
            pg.draw.polygon(self.screen, car.get("color", (220, 40, 40)), pts)
            pg.draw.line(self.screen, (255, 255, 255), self.to_px(car["x"], car["y"]),
                         self.to_px(car["x"] + c * hl, car["y"] + s * hl), 2)  # nose
        if text:
            self.screen.blit(self.font.render(text, True, (255, 255, 255)), (10, 8))
        pg.display.flip()
        return True

    def close(self):
        self.pg.quit()
