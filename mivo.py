""" An attempt at a vector DB with MiVo (MItotic VOronoi) clusters.

Idea
- Store points in files corresponding to centroids
- Crowded cells undergo mitosis along first principal axis
- Resummary, temporal decay, and k-means retraining go hand-in-hand
"""

from __future__ import annotations
from typing import (
    Generic as Of,
    Hashable,
    Literal as Lit,
    Protocol as Sig,
    TypeAlias as Typ,
    TypeVar as Tyvar,
)
from dataclasses import dataclass
from numpy.typing import NDArray
from utils import BiIndex, Sediment

import numpy as np

TagStr: Typ = str
ArrF16: Typ = NDArray[np.float16]

_H = Tyvar('_H', bound=Hashable)
_H_Con = Tyvar('_H_Con', bound=Hashable, contravariant=True)
_X = Tyvar('_X')

@dataclass
class Cluster(Of[_H]):
    row_id: BiIndex[int, _H]
    row_vec: Sediment[ArrF16]
    row_wgt: Sediment[ArrF16]
    dirty: bool  # this bit is for parent object

    def assert_consistent(self) -> None:
        g = self.row_id; X = self.row_vec.merge(); w = self.row_wgt.merge()
        assert X.ndim == 2
        assert w.ndim == 1
        n, _ = X.shape
        n_, = w.shape
        assert n == n_
        assert n == len(g)
        assert all(k in g for k in range(0, n))
        assert np.all(w >= 0)

    def __len__(self) -> int:
        return len(self.row_id)

    def compact(self) -> None:
        X = self.row_vec.merge(); w = self.row_wgt.merge()
        ne = (w != 0)
        self.row_id = BiIndex.from_mask_iter(ne).compose(self.row_id)
        self.row_vec = self.row_vec.new_like([X[ne]])
        self.row_wgt = self.row_wgt.new_like([w[ne]])
        self.dirty = True

    def center(
            self, *,
            assume_compact: bool=False,
            spherical: bool=False,
            ) -> ArrF16:
        if not assume_compact:
            self.compact()
        assert len(self) >= 1
        X = self.row_vec.frames[0]; w = self.row_wgt.frames[0]
        col = lambda v: v[:, None]
        c = (X * col(w)).sum(axis=0)
        return (c / np.linalg.norm(c)) if spherical else (c / w.sum())

    def mitosis(
            self, *,
            assume_compact: bool=False,
            spherical: bool=False,
            ) -> tuple[Cluster[_H], Cluster[_H]]:

        # check that the cluster can be split
        if not assume_compact:
            self.compact()
        assert len(self) >= 2
        X = self.row_vec.frames[0]; w = self.row_wgt.frames[0]
        row = lambda v: v[None, :]
        col = lambda v: v[:, None]

        # center the cluster
        c = (X * col(w)).sum(axis=0)
        if spherical:
            c /= np.linalg.norm(c)
            A = X - col(X @ c) * row(c)
        else:
            c /= w.sum()
            A = X - row(c)

        # perform weighted PCA
        _, _, vt = np.linalg.svd(A * np.sqrt(col(w)), full_matrices=False)
        axis = vt[0]

        # split along axis
        proj = A @ axis
        j = (proj < 0); k = ~j

        # filter
        c1 = Cluster(
            self.row_id.mask(j),
            self.row_vec.new_like([X[j]]),
            self.row_wgt.new_like([w[j]]),
            dirty=True,
        )
        c2 = Cluster(
            self.row_id.mask(k),
            self.row_vec.new_like([X[k]]),
            self.row_wgt.new_like([w[k]]),
            dirty=True,
        )
        return c1, c2

    def assign(
            self,
            g: _H,
            e: ArrF16,
            w: float, *,
            extant: bool | None=None,
            ) -> None:
        ex = (g in self.row_id.T)
        if extant is not None and ex != extant:
            raise IndexError(
                f"global index {g=} unexpected state of existence ({extant})")
        if ex:
            i = self.row_id.T[g]
            j, k = self.row_vec.i2pair(i)
            assert (j, k) == self.row_wgt.i2pair(i)
            self.row_vec.frames[j][k] = e
            self.row_wgt.frames[j][k] = w
            self.row_vec.soft_merge()
            self.row_wgt.soft_merge()
        else:
            self.row_id[len(self.row_id)] = g
            # self.mat_r = np.concatenate([self.mat_r, e[None, :]], axis=0)
            self.row_vec.extend(e[None, :])
            self.row_wgt.extend(np.array([w], dtype=np.float16))
        self.dirty=True

    def remove(self, g: _H) -> None:
        if g not in self.row_id.T:
            raise IndexError(f"global index {g=} doesn't exist here.")
        i = self.row_id.T[g]
        j, k = self.row_wgt.i2pair(i)
        self.row_wgt.frames[j][k] = 0
        self.dirty=True

    def l2sq_rank(self, q: ArrF16) -> list[tuple[_H, float, float]]:
        """ Given a query vector, return a lis tof IDs, scores, and weights.

        Note: the resulting scores are squared distances.
        """
        X_ = self.row_vec; X_.soft_merge()
        w_ = self.row_wgt; w_.soft_merge()
        scores = []
        for X in X_.frames:
            for i, s in enumerate(((X - q[None, :])**2).sum(axis=1)):
                j, k = w_.i2pair(i)
                w = w_.frames[j][k]
                if w != 0: scores.append((i, s, w))
        return scores

    def cosine_rank(
            self,
            q: ArrF16, *,
            spherical: bool=False,
            ) -> list[tuple[_H, float, float]]:
        """ Given a query vector, return a list of IDs, scores, and weights.

        If spherical, doesn't do extra normalization for cosine score.
        """
        X_ = self.row_vec; X_.soft_merge()
        w_ = self.row_wgt; w_.soft_merge()
        scores = []
        for X in X_.frames:
            raw_scores = X @ q
            if spherical:
                denom = (X**2).sum(axis=1).sqrt()
                raw_scores /= denom
                raw_scores /= (q**2).sum()
            for i, s in enumerate(raw_scores):
                j, k = w_.i2pair(i)
                w = w_.frames[j][k]
                if w != 0: scores.append((i, s, w))
        return scores

def least_sufficient(
        scores: list[tuple[_H, float, float]], *,
        beta: float,
        theta: float,
        scoring: Lit["l2sq", "cosine"],
        weighing: Lit["pre", "post"],
        ) -> list[tuple[_H, float, float]]:
    """ Given IDs, scores, and weights; apply soft clustering to find the least
    amount of most relevant points to account for a minimum threshold of total
    probability.

    - Dot product to distance: (x-q)² = x² + q² - 2x⋅q
    - Cosine to distance (normalized vecs): (x-q)² = 2 - 2cos(x,q)
    - Soft clustering: probability of q belonging to x is p = exp(-β(x-q)²)/Z
    - Weighing: we try to make weight w behave like w points of weight 1
    - pre-W soft clustering: p = w exp(-β(x-q)²) / Z
    - pre-W filtering: least j for which sum[i,0≤i<j] p[i] ≥ θ
    - post-W soft clustering: p = exp(-β(x-q)²) / Z
    - post-W filtering: least j for which sum[i,0≤i<j] w[i] p[i] ≥ θ

    I have no proof for why this should work. Just a hunch.
    """
    assert 0 <= theta <= 1
    scores = sorted(scores, key=lambda t: t[1], reverse=(scoring == "cosine"))
    match scoring:
        case "l2sq":
            l2sq = np.array([s for _, s, _ in scores])
        case "cosine":
            l2sq = np.array([2 - 2 * s for _, s, _ in scores])
    w = np.array([w for _, _, w in scores])
    match weighing:
        case "pre":
            h = w * np.exp(-beta * l2sq); h /= h.sum()
        case "post":
            h = np.exp(-beta * l2sq); h /= h.sum(); h *= w
    H = h.cumsum()
    ix = np.arange(len(scores))
    j = ix[H >= theta][0]
    return scores[:j + 1]

class ClusterLoader(Of[_H], Sig):
    def __getitem__(self, k: _H) -> Cluster[_H]:
        """ This can have side-effects such as unloading unused clusters,
        updating in-memory entries, loading corresponding RAG contents, etc.
        """
        ...
    def __setitem__(self, k: _H, v: Cluster[_H]) -> None:
        """ This method should not assign-by-reference, and should accept new
        keys.
        """
        ...
    def __delitem__(self, k: _H) -> None:
        ...

class MappedLoader(Of[_H_Con, _X], Sig):
    def __getitem__(self, k: _H_Con) -> _X: ...
    def __setitem__(self, k: _H_Con, aux: _X) -> None: ...
    def __delitem__(self, k: _H_Con) -> None: ...

class MiVoStorage(Of[_H, _X], Sig):
    def depth(self, k: _H) -> int: ...
    @property
    def height(self) -> int: ...

    @property
    def branch(self) -> ClusterLoader[_H]: ...
    @property
    def leaf(self) -> MappedLoader[_H, _X]: ...
    @property
    def root(self) -> _H: ...
    @root.setter
    def set_root(self, root: _H) -> None: ...

    def suggest_id(self) -> _H: ...
    def save(self) -> None: ...

@dataclass
class Threshold:
    weighing: Lit["pre", "post"]
    theta: float

class Thresholding(Sig):
    def __call__(
            self,
            d0: int, h: int,
            n: int, N: int, w: float,
            k: int,
            ) -> Threshold:
        """
        :param d0: zero-indexed depth of node
        :param h: MiVoTree height
        :param n: node children count
        :param N: node leaf count
        :param w: node weight
        :param k: number of wanted results
        :return: threshold to use for query
        """
        ...

class Splitting(Sig):
    def __call__(
            self,
            d0: int, h: int,
            n: int, N: int, w: float,
            ) -> bool:
        """
        :param d0: zero-indexed depth of node
        :param h: MiVoTree height
        :param n: node children count
        :param N: node leaf count
        :param w: node weight
        :return: whether to undergo mitosis
        """
        ...

@dataclass
class MiVoTree(Of[_H, _X]):
    """ The MiVoTree is a perfectly balanced multitree, similar in structural
    spirit to a 2-3-4-tree; however, it is also a leafy tree, and the branching
    nodes serve merely as navigational aids. As such, it is expected that all
    the actual data live on depth (height - 1).

    Tree pruning and shrinking is yet to be designed and implemented.
    """
    store: MiVoStorage[_H, _X]
    beta: float
    theta: float
    spherical: bool
    thresholding: Thresholding
    splitting: Splitting
