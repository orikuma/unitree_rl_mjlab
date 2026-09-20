"""Exact box geometry for a straight, five-step stair curriculum."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)


@dataclass(frozen=True)
class StairStage:
  """Physical dimensions of a curriculum row (all lengths in metres)."""

  num_steps: int
  height: float
  depth: float


# Increase height before shortening the treads; every non-flat row has five
# risers. The final row is explicitly 17 x 17 cm, without interpolation.
STAIR_STAGES = (
  StairStage(0, 0.00, 0.30),
  StairStage(5, 0.02, 0.30),
  StairStage(5, 0.04, 0.30),
  StairStage(5, 0.06, 0.30),
  StairStage(5, 0.08, 0.30),
  StairStage(5, 0.10, 0.30),
  StairStage(5, 0.12, 0.30),
  StairStage(5, 0.14, 0.30),
  StairStage(5, 0.17, 0.30),
  StairStage(5, 0.17, 0.25),
  StairStage(5, 0.17, 0.21),
  StairStage(5, 0.17, 0.19),
  StairStage(5, 0.17, 0.17),
)
TARGET_LEVEL = len(STAIR_STAGES) - 1

# X coordinates used by the MDP are relative to TerrainOutput.origin. The
# origin stays on the lower flat, including when the tread depth changes.
FIRST_RISER_X = 0.6
STAIR_WIDTH = 1.2
LANDING_LENGTH = 1.2  # Minimum clear landing; geometry extends to the patch end.
TERRAIN_SIZE = (5.5, 3.0)


@dataclass(kw_only=True)
class BoxFiveStepStairsTerrainCfg(SubTerrainCfg):
  """A lower flat, five full-height boxes, and a level upper landing.

  Use ``num_rows=len(stages)`` and ``difficulty_range=(0, 1)`` with a
  curriculum terrain generator. Each row then selects exactly one stage,
  despite the generator's random fractional difficulty within each row.
  The last tread begins at ``FIRST_RISER_X + 4 * depth`` and the landing at
  ``FIRST_RISER_X + 5 * depth``, relative to the returned spawn origin.
  """

  size: tuple[float, float] = TERRAIN_SIZE
  stages: tuple[StairStage, ...] = STAIR_STAGES
  stair_width: float = STAIR_WIDTH
  first_riser_x: float = FIRST_RISER_X
  # Include the reset jitter, torso IMU offset and rear half of the scan grid
  # inside this patch, so its height map cannot see the previous row's landing.
  spawn_x: float = 1.2
  minimum_landing_length: float = LANDING_LENGTH
  ground_thickness: float = 0.1

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    dimensions = (
      *self.size,
      self.stair_width,
      self.first_riser_x,
      self.spawn_x,
      self.minimum_landing_length,
      self.ground_thickness,
    )
    if not all(math.isfinite(value) and value > 0 for value in dimensions):
      raise ValueError("Terrain dimensions must be finite and positive.")
    if not self.stages:
      raise ValueError("At least one stair stage is required.")
    if not math.isfinite(difficulty):
      raise ValueError("Terrain difficulty must be finite.")
    if self.stair_width > self.size[1]:
      raise ValueError("The staircase must fit inside the terrain width.")
    for stage in self.stages:
      if (
        stage.num_steps not in (0, 5)
        or not math.isfinite(stage.height)
        or not math.isfinite(stage.depth)
        or stage.height < 0
        or stage.depth <= 0
        or (stage.num_steps == 0) != (stage.height == 0)
      ):
        raise ValueError("Each stage must be flat or have five positive risers.")
      required_length = (
        self.spawn_x
        + self.first_riser_x
        + stage.num_steps * stage.depth
        + self.minimum_landing_length
      )
      if required_length > self.size[0]:
        raise ValueError("Terrain length must include the approach and top landing.")

    level = min(max(int(difficulty * len(self.stages)), 0), len(self.stages) - 1)
    stage = self.stages[level]
    body = spec.body("terrain")
    center_y = self.size[1] / 2
    geometries: list[TerrainGeometry] = []

    def add_box(
      size: tuple[float, float, float],
      pos: tuple[float, float, float],
      color: tuple[float, float, float, float],
    ) -> None:
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        group=0,
      )
      geometries.append(TerrainGeometry(geom=geom, color=color))

    # All boxes meet only at their boundary surfaces. The ground spans z <= 0;
    # adjacent treads and the landing occupy separate x intervals at z >= 0.
    add_box(
      (self.size[0] / 2, self.size[1] / 2, self.ground_thickness / 2),
      (self.size[0] / 2, center_y, -self.ground_thickness / 2),
      (0.25, 0.28, 0.30, 1.0),
    )
    origin = np.array((self.spawn_x, center_y, 0.0))
    if stage.num_steps == 0:
      return TerrainOutput(origin=origin, geometries=geometries)

    stair_start_x = self.spawn_x + self.first_riser_x
    for step in range(1, stage.num_steps + 1):
      height = step * stage.height
      add_box(
        (stage.depth / 2, self.stair_width / 2, height / 2),
        (stair_start_x + (step - 0.5) * stage.depth, center_y, height / 2),
        (0.15, 0.35 + 0.08 * step, 0.55, 1.0),
      )

    landing_start_x = stair_start_x + stage.num_steps * stage.depth
    landing_length = self.size[0] - landing_start_x
    top_height = stage.num_steps * stage.height
    add_box(
      (landing_length / 2, self.stair_width / 2, top_height / 2),
      (landing_start_x + landing_length / 2, center_y, top_height / 2),
      (0.15, 0.75, 0.55, 1.0),
    )
    return TerrainOutput(origin=origin, geometries=geometries)
