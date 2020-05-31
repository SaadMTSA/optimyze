import numpy as np

from pyDOE import lhs
from sklearn.model_selection import cross_validate
from sklearn.base import clone


class LatinHypercubesCV:
    """
    LatinHypercubes Cross-Validation
    """

    def __init__(
        self,
        estimator,
        param_grid,
        n_iters=10,
        scoring=None,
        n_jobs=None,
        cv=None,
        refit=False,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_iters = n_iters
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit

    def _calculate_intervals(self):
        adjusted_hp_set = []
        for k, v in self.param_grid.items():
            adjusted_hp_set.append([])
            if isinstance(v, list):
                for i in self.hp_set[0] * len(v):
                    adjusted_hp_set[-1].append(v[int(i)])
            elif isinstance(v, tuple):
                if 2 <= len(v) <= 3:
                    possibilities = np.arange(*v)
                    for i in (
                        self.hp_set[0] * (v[1] - v[0]) / (1 if len(v) == 2 else v[2])
                    ):
                        adjusted_hp_set[-1].append(possibilities[int(i)])
                else:
                    raise ValueError(
                        "Param value of type tuple should be of "
                        "length 2 or 3 (start, end[, step])"
                    )
            else:
                raise TypeError(
                    f"Cannot use {type(v)} ({v}) as param grid value. "
                    "It should be a list or a tuple."
                )
            self.hp_set = self.hp_set[1:]

        self.hp_set = adjusted_hp_set

    def fit(self, X, y, groups=None, **kwargs):

        self.hp_set = lhs(len(self.param_grid), self.n_iters).T
        self._calculate_intervals()
        self.scores = []

        self.best_score_ = 0
        self.best_index_ = 0
        self.best_estimator_ = None
        self.best_params_ = None

        for i in range(self.n_iters):
            current_params = {}
            for j_idx, j in enumerate(self.param_grid.keys()):
                current_params[j] = self.hp_set[j_idx][i]
            self.estimator.set_params(**current_params)
            self.scores.append(
                cross_validate(
                    self.estimator,
                    X=X,
                    y=y,
                    groups=groups,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    **kwargs,
                )
            )
            current_score = np.mean(self.scores[-1]["test_score"])
            if self.best_score_ < current_score:
                self.best_score_ = current_score
                self.best_params_ = current_params
                self.best_index = np.size(self.scores) - 1
                self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.fit(X, y, groups, **kwargs)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)
