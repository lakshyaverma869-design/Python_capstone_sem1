"""Canonical Erangel coordinates and a landmark-calibrated image transform."""
from __future__ import annotations

from dataclasses import dataclass

REFERENCE_WIDTH = 1181.0
REFERENCE_HEIGHT = 1133.0
REFERENCE_SHORT_SIDE = min(REFERENCE_WIDTH, REFERENCE_HEIGHT)
COORDINATE_SCALE = 1000.0 / REFERENCE_SHORT_SIDE
MAP_MIN = 0.0
MAP_WIDTH = REFERENCE_WIDTH * COORDINATE_SCALE
MAP_HEIGHT = REFERENCE_HEIGHT * COORDINATE_SCALE
MAP_SHORT_SIDE = 1000.0

# Axis-aligned least-squares fit from 22 corresponding named Erangel locations.
# Coordinates are source-reference label/location centers -> uploaded map pixels.
# Three text anchors with inconsistent label placement were excluded.
SOURCE_TO_IMAGE_X_SCALE = 0.41899914
SOURCE_TO_IMAGE_X_OFFSET = 0.72323129
SOURCE_TO_IMAGE_Y_SCALE = 0.40657498
SOURCE_TO_IMAGE_Y_OFFSET = -6.69447652
LANDMARK_RMS_ERROR_PX = 15.32286193
LANDMARK_MAX_ERROR_PX = 25.72705190
LANDMARK_COUNT = 22

# Each tuple: (name, source-reference x, source-reference y, uploaded-image x, uploaded-image y)
LANDMARK_PAIRS = (
    ("Zharki",130,188,72,59), ("Severny",536,175,245,73), ("Stalber",815,186,329,59),
    ("Kameshki",955,151,423,58), ("Georgopol",242,378,89,147), ("Yasnaya Polyana",763,351,315,132),
    ("Rozhok",560,422,236,161), ("Ruins",436,497,171,196), ("School",594,494,247,202),
    ("Mansion",889,458,358,184), ("Lipovka",1014,484,435,193), ("Gatka",294,613,123,235),
    ("Pochinki",491,599,209,250), ("Farm",760,675,303,268), ("Shelter",806,582,344,232),
    ("Prison",892,574,387,217), ("Mylta",838,698,350,267), ("Mylta Power",1035,640,436,277),
    ("Quarry",136,794,75,318), ("Ferry Pier",372,854,145,337),
    ("Novorepnoye",879,896,379,349), ("Sosnovka Military Base",642,1014,244,405),
)

@dataclass(frozen=True)
class MapCoordinateTransform:
    image_width: float = 474.0
    image_height: float = 429.0
    x_scale: float = SOURCE_TO_IMAGE_X_SCALE / COORDINATE_SCALE
    x_offset: float = SOURCE_TO_IMAGE_X_OFFSET
    y_scale: float = SOURCE_TO_IMAGE_Y_SCALE / COORDINATE_SCALE
    y_offset: float = SOURCE_TO_IMAGE_Y_OFFSET

    def contains(self, x: float, y: float) -> bool:
        return 0.0 <= x <= MAP_WIDTH and 0.0 <= y <= MAP_HEIGHT

    def model_to_display(self, x: float, y: float) -> tuple[float, float]:
        return self.x_scale * float(x) + self.x_offset, self.y_scale * float(y) + self.y_offset

    def display_to_model(self, x: float, y: float) -> tuple[float, float]:
        return ((float(x) - self.x_offset) / self.x_scale,
                (float(y) - self.y_offset) / self.y_scale)

    def pixel_to_model(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        if x < 0 or x > width or y < 0 or y > height:
            raise ValueError("Point falls outside the playable image bounds")
        image_x = float(x) * self.image_width / width
        image_y = float(y) * self.image_height / height
        result = self.display_to_model(image_x, image_y)
        if not self.contains(*result):
            raise ValueError("Point falls outside the calibrated map bounds")
        return result

    def model_to_pixel(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        if not self.contains(x, y):
            raise ValueError("Model point falls outside the configured map bounds")
        image_x, image_y = self.model_to_display(x, y)
        return image_x * width / self.image_width, image_y * height / self.image_height

    def model_radius_to_ellipse(self, radius: float, width: float | None = None,
                                height: float | None = None) -> tuple[float, float]:
        width = self.image_width if width is None else width
        height = self.image_height if height is None else height
        if width <= 0 or height <= 0 or radius < 0:
            raise ValueError("Image dimensions must be positive and radius non-negative")
        return (float(radius) * self.x_scale * width / self.image_width,
                float(radius) * self.y_scale * height / self.image_height)

    def model_radius_to_pixel(self, radius: float, width: float | None = None,
                              height: float | None = None) -> float:
        # Area-equivalent radius for metrics only. Render circles through the full affine transform.
        rx, ry = self.model_radius_to_ellipse(radius, width, height)
        return (rx * ry) ** 0.5

    def landmark_errors(self) -> list[tuple[str, float]]:
        errors = []
        for name, sx, sy, tx, ty in LANDMARK_PAIRS:
            x, y = self.model_to_display(sx * COORDINATE_SCALE, sy * COORDINATE_SCALE)
            errors.append((name, ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5))
        return errors
