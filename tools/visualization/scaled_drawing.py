"""Draw logical pixel coordinates directly onto a higher-resolution canvas."""
from PIL import ImageDraw


class ScaledDraw:
    def __init__(self, image, scale=1):
        if scale not in (1, 2):
            raise ValueError("drawing scale must be 1 or 2")
        self.draw, self.scale = ImageDraw.Draw(image), scale

    def coords(self, values):
        return tuple(self.coords(value) if isinstance(value, (tuple, list))
                     else value * self.scale for value in values)

    def text(self, xy, text, **kwargs):
        # Fonts are created at native size by the caller, never resized rasters.
        self.draw.text(self.coords(xy), text, **kwargs)

    def rectangle(self, xy, *, width=1, **kwargs):
        self.draw.rectangle(self.coords(xy), width=width * self.scale, **kwargs)

    def rounded_rectangle(self, xy, *, radius=0, width=1, **kwargs):
        self.draw.rounded_rectangle(self.coords(xy), radius=radius * self.scale,
                                    width=width * self.scale, **kwargs)

    def line(self, xy, *, width=1, **kwargs):
        self.draw.line(self.coords(xy), width=width * self.scale, **kwargs)
