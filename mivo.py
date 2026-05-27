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

    def mitosis(
            self, *,
            spherical: bool=False,
            ) -> tuple[Cluster[_H], Cluster[_H]]:

        # check that the cluster can be split
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

class LoadsCluster(Of[_H], Sig):
    @property
    def cluster(self) -> Cluster[_H]:
        """ This can have side-effects such as unloading unused clusters,
        updating in-memory entries, loading corresponding RAG contents, etc.
        """
        ...
    def save(self, self_id: _H) -> None:
        """ This should ensure that the corresponding cluster is saved
        correctly to a persistent storage, provided that the self_id is not in
        conflict with an existing one.
        """
        ...

def threshold(
        scores: list[tuple[_H, float, float]], *,
        beta: float,
        theta: float,
        scoring: Lit["l2sq", "cosine"],
        summing: Lit["prob", "entr"],
        ) -> list[tuple[_H, float, float]]:
    """ Given IDs, scores, and weights; apply soft clustering to find the least
    amount of most relevant points to account for a minimum threshold of total
    probability/entropy.

    - Dot product to distance: (x-q)² = x² + q² - 2x⋅q
    - Cosine to distance (normalized vecs): (x-q)² = 2 - 2cos(x,q)
    - Soft clustering: probability of q belonging to x is p = exp(-β(x-q)²)/Z
    - Entropy: given by S = sum[i] p[i](-ln(p[i]))
    - Weighing: we try to make weight w behave like w points of weight 1
    - W-Soft clustering: p = w exp(-β(x-q)²) / Z
    - W-Entropy: S[j] = sum[i,0≤i<j] p[i]( -ln(p[i] / w[i]) )
    - Filtering for probability: least j for which sum[i,0≤i<j] p[i] ≥ θ
    - Filtering for entropy: least j for which S[j]/S ≥ θ

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
    h = w * np.exp(-beta * l2sq); h /= h.sum()
    if summing == "entr":
        h = h * np.log(w / h); h /= h.sum()
    H = h.cumsum()
    ix = np.arange(len(scores))
    j = ix[H >= theta][0]
    return scores[:j + 1]
