"""Minimal GGUF weight access: dequantised float64 matrices in (rows=out, cols=in) orientation."""
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


class Weights:
    def __init__(self, path):
        self.reader = GGUFReader(path, "r")
        self.by_name = {t.name: t for t in self.reader.tensors}

    def get(self, name):
        t = self.by_name[name]
        arr = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float64)
        ne = [int(x) for x in t.shape]            # ggml order: ne[0] = innermost (input) dim
        if len(ne) == 1:
            return arr.reshape(ne[0])
        return np.ascontiguousarray(arr.reshape(ne[1], ne[0]))  # (out, in)

    def get_rows(self, name, rows):
        """Dequantise only selected rows of a 2D tensor (used for the 248320-row embedding)."""
        t = self.by_name[name]
        ne = [int(x) for x in t.shape]
        data = t.data
        if data.shape[0] != ne[1]:
            raise RuntimeError(f"unexpected raw layout {data.shape} for {name} with ne={ne}")
        sub = np.ascontiguousarray(data[np.asarray(rows, dtype=np.int64)])
        return np.asarray(dequantize(sub, t.tensor_type), dtype=np.float64).reshape(len(rows), ne[0])

    def qtype(self, name):
        return self.by_name[name].tensor_type.name
