# Adapted from concepts by:
# Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System',
#   Proceedings of the 22nd ACM SIGKDD, pp. 785-794.
# Ke, G. et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
#   Advances in Neural Information Processing Systems 30.
# He, H. and Garcia, E. A. (2009) 'Learning from Imbalanced Data',
#   IEEE Transactions on Knowledge and Data Engineering, 21(9), pp. 1263-1284.

"""Thin wrapper around XGBoost (Chen and Guestrin, 2016) for the AI-NIDS heads.

The same wrapper is instantiated three times — once for each ablation
configuration — with different feature sets: hand-crafted session
statistics only, Transformer embeddings only, or the concatenation of the
two. Training uses the histogram tree method whose ideas trace back to
LightGBM (Ke et al., 2017); class imbalance is handled with log-smoothed
inverse-frequency sample weights, following the general recommendations of
He and Garcia (2009).
"""
import logging
import os
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class ZeekXGBoost:
    """
    Thin wrapper around XGBClassifier with serialisation helpers.

    Parameters
    ----------
    cfg : dict
        Config dict (top-level project config); reads cfg['xgboost'].
    num_classes : int
    """

    def __init__(self, cfg: dict | None = None, num_classes: int = 15):
        self.num_classes = num_classes
        xcfg = (cfg or {}).get("xgboost", {})

        # Hyperparameters are the ones selected by offline validation on
        # the CICIDS2017 training split; histogram tree method is used
        # throughout (Chen and Guestrin, 2016; Ke et al., 2017).
        self.model = xgb.XGBClassifier(
            n_estimators=xcfg.get("n_estimators", 300),
            max_depth=xcfg.get("max_depth", 8),
            learning_rate=xcfg.get("learning_rate", 0.1),
            subsample=xcfg.get("subsample", 0.8),
            colsample_bytree=xcfg.get("colsample_bytree", 0.8),
            min_child_weight=xcfg.get("min_child_weight", 3),
            gamma=xcfg.get("gamma", 0.1),
            reg_alpha=xcfg.get("reg_alpha", 0.1),
            reg_lambda=xcfg.get("reg_lambda", 1.0),
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda" if xcfg.get("use_gpu", False) else "cpu",
            random_state=42,
            verbosity=0,
        )
        self._fitted = False
        self._feature_names: list[str] = []
        self._label_encoder: LabelEncoder | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 20,
    ) -> "ZeekXGBoost":
        if feature_names:
            self._feature_names = feature_names

        self._label_encoder = LabelEncoder()
        y_train_enc = self._label_encoder.fit_transform(y_train)
        # XGBoost expects classes to be contiguous 0..K-1.
        self.model.set_params(num_class=len(self._label_encoder.classes_))

        eval_set = []
        if X_val is not None and y_val is not None:
            unseen_in_val = np.setdiff1d(np.unique(y_val), self._label_encoder.classes_)
            if unseen_in_val.size == 0:
                y_val_enc = self._label_encoder.transform(y_val)
                eval_set = [(X_val, y_val_enc)]
            else:
                logger.warning(
                    "Validation set contains unseen classes %s; disabling eval_set for this fit.",
                    unseen_in_val.tolist(),
                )

        if eval_set and early_stopping_rounds:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

        self.model.fit(
            X_train, y_train_enc,
            eval_set=eval_set if eval_set else None,
            verbose=False,
            sample_weight=sample_weight,
        )
        self._fitted = True
        logger.info("XGBoost training complete. Best iteration: %s",
                    getattr(self.model, "best_iteration", "N/A"))
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_enc = self.model.predict(X).astype(np.int64)
        if self._label_encoder is None:
            return pred_enc
        return self._label_encoder.inverse_transform(pred_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_enc = self.model.predict_proba(X)
        if self._label_encoder is None:
            return proba_enc

        classes = self._label_encoder.classes_.astype(np.int64)
        if len(classes) == self.num_classes:
            return proba_enc

        proba_full = np.zeros((proba_enc.shape[0], self.num_classes), dtype=proba_enc.dtype)
        proba_full[:, classes] = proba_enc
        return proba_full

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model,
                     "feature_names": self._feature_names,
                     "num_classes": self.num_classes,
                     "label_classes": self._label_encoder.classes_.tolist() if self._label_encoder is not None else None}, path)
        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ZeekXGBoost":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj._feature_names = data.get("feature_names", [])
        obj.num_classes = data.get("num_classes", 15)
        label_classes = data.get("label_classes")
        obj._label_encoder = None
        if label_classes is not None:
            obj._label_encoder = LabelEncoder()
            obj._label_encoder.classes_ = np.array(label_classes)
        obj._fitted = True
        return obj

    # ── Reporting ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        label_names: Optional[list[str]] = None,
    ) -> dict:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        y_pred = self.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
        logger.info("XGBoost eval  acc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f", acc, f1, prec, rec)
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "report": report}

    @property
    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Return per-sample weights that balance the training distribution.

    A raw inverse-frequency weighting produces weights in the thousands for
    classes with single-digit support in CICIDS2017, which silently breaks
    XGBoost's ``min_child_weight`` constraint and caused the pathological
    all-one-class prediction bug observed during the first training run.
    The log-smoothed form ``log(N / count_k) + 1`` flattens that tail while
    keeping the relative ordering of class importance, and weights are
    finally renormalised so their sum equals ``len(y)`` — this keeps the
    Hessian-based split thresholds on the same scale as the default
    unweighted case.
    """
    counts = np.bincount(y, minlength=num_classes).astype(float)
    present = counts > 0
    n_samples = float(len(y))

    weights_per_class = np.zeros(num_classes, dtype=float)
    weights_per_class[present] = np.log(n_samples / counts[present]) + 1.0

    sample_weights = weights_per_class[y]
    # Normalise so sum(weights) == n_samples
    sample_weights *= n_samples / sample_weights.sum()
    return sample_weights
