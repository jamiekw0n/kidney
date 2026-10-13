"""Standalone bounded TIFF reads and zero-padded RGB context crops."""
import numpy as np


class TiffRegionReader:
    def __init__(self, path, expected_channels, axes=None, series=0, level=0):
        import tifffile
        import zarr
        self.tif = tifffile.TiffFile(path, mode='r')
        self.store = None
        try:
            selected = self.tif.series[series].levels[level]
            self.axes = axes or selected.axes
            if len(self.axes) != len(selected.shape) or len(set(self.axes)) != len(self.axes):
                raise ValueError('Axes must uniquely label every TIFF dimension')
            channel_axes = [a for a in 'CS' if a in self.axes]
            if 'Y' not in self.axes or 'X' not in self.axes or len(channel_axes) != (0 if expected_channels is None else 1):
                raise ValueError(f'Expected YX and exactly one C/S channel axis, got {self.axes}')
            self.channel_axis = channel_axes[0] if channel_axes else None
            for axis, size in zip(self.axes, selected.shape):
                if axis not in 'YXCS' and size != 1:
                    raise ValueError(f'Non-singleton {axis}: export selected plane first')
            if expected_channels is not None and selected.shape[self.axes.index(self.channel_axis)] != expected_channels:
                raise ValueError('Channel count does not match expected panel')
            self.height = selected.shape[self.axes.index('Y')]
            self.width = selected.shape[self.axes.index('X')]
            self.dtype = selected.dtype
            self.store = selected.aszarr()
            self.array = zarr.open(self.store, mode='r')
            if not hasattr(self.array, 'shape'):
                self.array = self.array['0']
            if tuple(self.array.shape) != tuple(selected.shape):
                raise ValueError('Unexpected TIFF pyramid level from Zarr adapter')
        except Exception:
            self.close()
            raise

    def read(self, y0, y1, x0, x1):
        if not (0 <= y0 <= y1 <= self.height and 0 <= x0 <= x1 <= self.width):
            raise ValueError('Read bounds outside image')
        selectors = [slice(y0,y1) if a == 'Y' else slice(x0,x1) if a == 'X'
                     else slice(None) if a == self.channel_axis else 0 for a in self.axes]
        data = np.asarray(self.array[tuple(selectors)])
        remaining = ''.join(a for a in self.axes if a in ('Y','X',self.channel_axis))
        order = ('Y','X') if self.channel_axis is None else ('Y','X',self.channel_axis)
        return data.transpose([remaining.index(a) for a in order])

    def close(self):
        if self.store is not None:
            self.store.close()
        self.tif.close()


def read_he_patch(reader, x, y, size=128):
    if size <= 0 or size % 2:
        raise ValueError('Context size must be positive and even')
    if not (0 <= x < reader.width and 0 <= y < reader.height):
        raise ValueError('Crop center outside image')
    half = size // 2
    patch = reader.read(max(0,y-half), min(reader.height,y+half),
                        max(0,x-half), min(reader.width,x+half))
    if patch.dtype != np.uint8 or patch.shape[-1] != 3:
        raise ValueError('H&E must be uint8 RGB')
    return np.pad(patch, ((max(0,half-y),max(0,y+half-reader.height)),
                         (max(0,half-x),max(0,x+half-reader.width)),(0,0)))

class CachedRegionReader:
    """Decode RGB once, then slice the exact native pixels. Original reader owns cleanup."""
    def __init__(self, reader):
        self.reader = reader
        self.pixels = reader.read(0, reader.height, 0, reader.width)
    def __getattr__(self, name):
        return getattr(self.reader, name)
    def read(self, y0, y1, x0, x1):
        if not (0 <= y0 <= y1 <= self.height and 0 <= x0 <= x1 <= self.width):
            raise ValueError('Read bounds outside image')
        return self.pixels[y0:y1, x0:x1]
    def close(self):
        self.reader.close()
