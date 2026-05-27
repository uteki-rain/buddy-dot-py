""" Utilities.
"""

from __future__ import annotations
from typing import (
    Callable as Fn,
    Generic as Of,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    TypeVar as Tyvar,
)
from dataclasses import dataclass

_H1 = Tyvar('_H1', bound=Hashable)
_H2 = Tyvar('_H2', bound=Hashable)
_H3 = Tyvar('_H3', bound=Hashable)
_D = Tyvar('_D')

@dataclass
class BiIndex(Of[_H1, _H2]):
    _l: dict[_H1, _H2]
    _r: dict[_H2, _H1]
    _transpose_cache: BiIndex[_H2, _H1] | None

    def __init__(self, d: Mapping[_H1, _H2] | None=None) -> None:
        self._l = {}
        self._r = {}
        if d is not None:
            for k in d:
                self[k] = d[k]

    @staticmethod
    def from_iter(it: Iterable[_H2]) -> BiIndex[int, _H2]:
        return BiIndex({i: j for i, j in enumerate(it)})
    @staticmethod
    def from_mask_map(mask: Mapping[_H2, bool]) -> BiIndex[int, _H2]:
        return BiIndex.from_iter(i for i in mask if mask[i])
    @staticmethod
    def from_mask_iter(it: Iterable[bool]) -> BiIndex[int, int]:
        return BiIndex.from_mask_map({i: j for i, j in enumerate(it)})

    @staticmethod
    def __mk_raw(l: dict[_H1, _H2], r: dict[_H2, _H1]) -> BiIndex[_H1, _H2]:
        b = BiIndex()
        b._l = l
        b._r = r
        return b

    @property
    def T(self) -> BiIndex[_H2, _H1]:
        if self._transpose_cache is None:
            self._transpose_cache = BiIndex.__mk_raw(self._r, self._l)
            self._transpose_cache._transpose_cache = self
        return self._transpose_cache

    def __len__(self) -> int:
        return len(self._l)
    def __iter__(self) -> Iterator[_H1]:
        return iter(self._l)
    def items(self) -> Iterator[tuple[_H1, _H2]]:
        for k, v in self._l.items(): yield (k, v)

    def __str__(self) -> str: return (
        "{" + ", ".join(f"{k!r} ↔ {v!r}" for k, v in self._l.items()) + "}")

    def __getitem__(self, k: _H1) -> _H2:
        return self._l[k]
    def get(self, k: _H1, v0: _D) -> _H2 | _D:
        return self._l.get(k, v0)
    def __setitem__(self, k: _H1, v: _H2) -> None:
        if k in self._l:
            raise IndexError(f"{k=} exists in BiIndex")
        if v in self._r:
            raise IndexError(f"{v=} exists in BiIndex")
        self._l[k] = v
        self._r[v] = k
    def __delitem__(self, k: _H1) -> None:
        if k in self._l:
            assert self._l[k] in self._r and self._r[self._l[k]] == k
            self._r.pop(self._l[k])
            self._l.pop(k)
        else:
            raise IndexError(f"{k=} doesn't exist in BiIndex")

    def mask(self, m: Mapping[_H1, bool]) -> BiIndex[_H1, _H2]:
        b = BiIndex()
        for k, v in self._l.items():
            if m[k]:
                b[k] = v
        return b
    def restrict(self, ks: Iterable[_H1]) -> BiIndex[_H1, _H2]:
        b = BiIndex()
        for k in ks:
            if k in self._l:
                b[k] = self._l[k]
        return b
    def compose(self, other: BiIndex[_H2, _H3]) -> BiIndex[_H1, _H3]:
        b = BiIndex()
        for v in set(*self.T).union(set(*other)):
            b[self.T[v]] = other[v]
        return b

@dataclass
class Sediment(Of[_D]):
    _size: Fn[[_D], int]
    _empty: _D
    _concat: Fn[[_D, _D], _D]
    _alpha: float
    frames: list[_D]

    def merge(self) -> _D:
        last = None
        buf = []
        while len(self.frames) > 1:
            for x in self.frames:
                if last is None:
                    last = (x,)
                else:
                    buf.append(self._concat(last[0], x))
                    last = None
            self.frames = buf
        return self.frames[0] if self.frames else self._empty

    def soft_merge(self) -> None:
        n = self._size; a = self._alpha; b = self.frames
        while len(b) >= 2 and n(b[-1]) * a >= n(b[-2]):
            x = b.pop(); w = b.pop()
            b.append(self._concat(w, x))

    def extend(self, x: _D) -> None:
        self.frames.append(x)
        self.soft_merge()

    def i2pair(self, i: int) -> tuple[int, int]:
        for j, x in enumerate(self.frames):
            k = self._size(x)
            if i < k: return (j, i)
            i -= k
        raise IndexError(f"out of bounds ({i!r})")

    def new_like(self, frames: list[_D]) -> Sediment[_D]:
        return Sediment(
            _size=self._size,
            _empty=self._empty,
            _concat=self._concat,
            _alpha=self._alpha,
            frames=frames,
        )
