"""Purpose-built terrain for five-step stair climbing."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)


@dataclass(kw_only=True)
class BoxFiveStepStairsTerrainCfg(SubTerrainCfg):
  """A straight staircase with a flat approach and top landing.

  Difficulty selects one entry from ``stages``. The staircase always ends at
  ``landing_start_x``, so the goal has the same position at every curriculum
  level even though tread depth changes.
  """

  stages: tuple[tuple[float, float], ...] = (
    (0.05, 0.30),
    (0.08, 0.28),
    (0.12, 0.25),
    (0.14, 0.22),
    (0.17, 0.17),
  )
  """Pairs of (step height, tread depth), in meters."""

  num_steps: int = 5
  stair_width: float = 1.2
  landing_start_x: float = 3.0
  spawn_x: float = 0.4
  ground_thickness: float = 0.1

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    if not self.stages:
      raise ValueError("At least one stair stage is required.")
    if self.num_steps <= 0:
      raise ValueError("num_steps must be positive.")
    if self.stair_width > self.size[1]:
      raise ValueError("stair_width must fit inside the terrain width.")
    if not 0.0 < self.spawn_x < self.landing_start_x < self.size[0]:
      raise ValueError("Expected spawn_x < landing_start_x < terrain length.")

    stage_index = min(int(difficulty * len(self.stages)), len(self.stages) - 1)
    step_height, step_depth = self.stages[stage_index]
    stair_start_x = self.landing_start_x - self.num_steps * step_depth
    if stair_start_x <= self.spawn_x:
      raise ValueError("The selected stage leaves no flat approach to the stairs.")

    body = spec.body("terrain")
    center_y = self.size[1] / 2.0
    geometries: list[TerrainGeometry] = []

    ground = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(
        self.size[0] / 2.0,
        self.size[1] / 2.0,
        self.ground_thickness / 2.0,
      ),
      pos=(self.size[0] / 2.0, center_y, -self.ground_thickness / 2.0),
    )
    geometries.append(
      TerrainGeometry(geom=ground, color=(0.25, 0.28, 0.30, 1.0))
    )

    for step_index in range(1, self.num_steps + 1):
      height = step_index * step_height
      center_x = stair_start_x + (step_index - 0.5) * step_depth
      step = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(step_depth / 2.0, self.stair_width / 2.0, height / 2.0),
        pos=(center_x, center_y, height / 2.0),
      )
      shade = 0.35 + 0.08 * step_index
      geometries.append(
        TerrainGeometry(geom=step, color=(0.15, shade, 0.55, 1.0))
      )

    top_height = self.num_steps * step_height
    landing_length = self.size[0] - self.landing_start_x
    landing = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(landing_length / 2.0, self.stair_width / 2.0, top_height / 2.0),
      pos=(
        self.landing_start_x + landing_length / 2.0,
        center_y,
        top_height / 2.0,
      ),
    )
    geometries.append(
      TerrainGeometry(geom=landing, color=(0.15, 0.75, 0.55, 1.0))
    )

    origin = np.array((self.spawn_x, center_y, 0.0))
    return TerrainOutput(origin=origin, geometries=geometries)
