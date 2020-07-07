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
        param_distributions,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        cv=None,
        refit=False,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.refit = refit
        self.best_score_ = 0
        self.best_index_ = 0
        self.best_estimator_ = None
        self.best_params_ = None

    def _calculate_intervals(self):
        adjusted_hp_set = []
        for k, v in self.param_distributions.items():
            adjusted_hp_set.append([])
            if isinstance(v, list):
                for i in self.hp_set[0] * len(v):
                    adjusted_hp_set[-1].append(v[int(i)])
            elif isinstance(v, tuple) and len(v) == 3:
                if v[0] == "uniform":
                    adjusted_hp_set[-1] = [i for i in (self.hp_set[0] * (v[2] - v[1]) + v[1])]
                elif v[0] == "log_uniform":
                    possibilities = np.logspace(
                        np.log2(v[1]), np.log2(v[2]), self.n_iter + 1, base=2
                    )
                    intervals = (self.hp_set[0] * self.n_iter).astype(np.int64)
                    for i in self.hp_set[0]:
                        interval = int(i * self.n_iter)
                        adjusted_hp_set[-1].append(
                            (possibilities[interval + 1] - possibilities[interval]) * i
                            + possibilities[interval]
                        )
                elif v[0] == "int_uniform":
                    adjusted_hp_set[-1] = [
                        int(i) for i in (self.hp_set[0] * (v[2] - v[1]) + v[1])
                    ]

            else:
                raise TypeError(
                    f"Cannot use {type(v)} ({v}) as param grid value. "
                    "It should be a list or a tuple of length 3."
                )
            self.hp_set = self.hp_set[1:]

        self.hp_set = adjusted_hp_set

    def _create_cv_results(self):
        cv_results = {}
        for i in self.scores:
            for j in i.keys():
                mean_str = f"mean_{j}"
                std_str = f"std_{j}"
                mean = np.mean(i[j])
                std = np.std(i[j])
                if mean_str in cv_results:
                    cv_results[mean_str].append(mean)
                else:
                    cv_results[mean_str] = [mean]

                if std_str in cv_results:
                    cv_results[std_str].append(std)
                else:
                    cv_results[std_str] = [std]

        for i in cv_results.keys():
            cv_results[i] = np.array(cv_results[i])

        self.cv_results_.update(cv_results)

    def fit(self, X, y, groups=None, **fit_params):

        self.cv_results_ = {"params": []}

        self.hp_set = lhs(len(self.param_distributions), self.n_iter).T
        self._calculate_intervals()
        scores = []

        for i in range(self.n_iter):
            current_params = {}
            for j_idx, j in enumerate(self.param_distributions.keys()):
                current_params[j] = self.hp_set[j_idx][i]

            self.estimator.set_params(**current_params)
            self.cv_results_["params"].append(current_params)

            scores.append(
                cross_validate(
                    self.estimator,
                    X=X,
                    y=y,
                    groups=groups,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    fit_params=fit_params,
                )
            )
            self.scores = scores
            current_score = np.mean(self.scores[-1]["test_score"])
            if self.best_score_ < current_score:
                self.best_score_ = current_score
                self.best_params_ = current_params
                self.best_index = np.size(self.scores) - 1
                self.best_estimator_ = clone(self.estimator)

        self._create_cv_results()

        self.best_estimator_.fit(X, y, **fit_params)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)
