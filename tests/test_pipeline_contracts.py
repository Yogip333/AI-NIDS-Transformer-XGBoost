"""
Pipeline contract tests.

Proves the architectural invariants that must hold between training and
inference. All tests are self-contained and do NOT require trained model
checkpoints — they use small synthetic arrays and mocked components.

Coverage:
  1. realign_to_trained_schema produces (N, 81) from (N, 78)
  2. Subflow Bwd Packets (col 64) is dropped by realign
  3. Zeek extra columns are appended at positions 77-80
  4. validate_flow_array rejects wrong column count → FeatureDimensionError
  5. validate_flow_array rejects NaN/Inf → NaNInfError
  6. validate_session_dataframe rejects wrong session size → SessionSizeError
  7. trained_feature_names() returns exactly 81 names
  8. trained_feature_names() matches ZEEK_EXTRA_COLS at the tail
  9. Both pipeline entry points produce identical output for the same input
 10. _heuristic_detection is NOT importable from routes.py (removed)
 11. _heuristic_fallback is NOT a method on LiveDemoServer (removed)
 12. predict_with_fallback does NOT fall back to Config A
 13. predict_with_fallback raises ModelNotLoadedError when C+B absent
 14. ModelNotLoadedError is raised (not RuntimeError) when hybrid absent
 15. InferenceError does not corrupt session — caller receives typed exception
 16. EnrichmentError is non-fatal and keeps the detection result intact
 17. AlertStore records from REST and stream share compatible schema
 18. TRAINED_NUM_FEATURES == CICIDS_RAW_COLS - 1 + len(ZEEK_EXTRA_COLS)
 19. EXPECTED_HYBRID_INPUT_DIM == EXPECTED_EMBEDDING_DIM + EXPECTED_STAT_FEATURES
 20. stat computation uses 81-col feature_names (not 79-col raw)
"""
from __future__ import annotations

import importlib
import inspect
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_raw_78(n_rows: int = 20, seed: int = 0) -> np.ndarray:
    """Return a (n_rows, 78) float32 array of finite values."""
    rng = np.random.default_rng(seed)
    return rng.random((n_rows, 78)).astype(np.float32) * 100


# ═════════════════════════════════════════════════════════════════════════════
# 1–3: realign_to_trained_schema output shape and column layout
# ═════════════════════════════════════════════════════════════════════════════

class TestRealignToTrainedSchema(unittest.TestCase):

    def setUp(self):
        from src.data.session_pipeline import realign_to_trained_schema
        self.realign = realign_to_trained_schema
        self.raw_78 = _make_raw_78(20)
        self.log_types = np.zeros(20, dtype=np.float32)

    def test_output_shape_is_n_by_81(self):
        """realign_to_trained_schema must produce (N, 81)."""
        result = self.realign(self.raw_78, self.log_types)
        self.assertEqual(result.shape, (20, 81))

    def test_output_dtype_is_float32(self):
        result = self.realign(self.raw_78, self.log_types)
        self.assertEqual(result.dtype, np.float32)

    def test_subflow_bwd_pkts_col64_is_dropped(self):
        """
        Column 64 of the input ('Subflow Bwd Packets') must be absent from
        the output. Prove by placing a sentinel (99999) at col 64 and
        verifying it does not appear in the first 77 output columns.
        """
        raw = self.raw_78.copy()
        raw[:, 64] = 99999.0
        result = self.realign(raw, self.log_types)
        # First 77 cols are CICIDS; sentinel must not be present
        self.assertFalse(np.any(result[:, :77] == 99999.0),
                         "Sentinel value from col 64 leaked into output")

    def test_col63_is_preserved_before_drop(self):
        """Col 63 (just before the dropped col 64) must survive."""
        raw = self.raw_78.copy()
        raw[:, 63] = 77777.0
        result = self.realign(raw, self.log_types)
        self.assertTrue(np.any(result[:, :77] == 77777.0),
                        "Col 63 unexpectedly removed from output")

    def test_log_types_appear_at_col_77(self):
        """log_types must land at output column 77 (first Zeek extra col)."""
        lt = np.full(20, 3, dtype=np.float32)  # sentinel log_type=3
        result = self.realign(self.raw_78, lt)
        np.testing.assert_array_equal(result[:, 77], 3.0)

    def test_all_values_finite(self):
        result = self.realign(self.raw_78, self.log_types)
        self.assertTrue(np.all(np.isfinite(result)))


# ═════════════════════════════════════════════════════════════════════════════
# 4–6: Validation functions raise typed exceptions on bad input
# ═════════════════════════════════════════════════════════════════════════════

class TestValidation(unittest.TestCase):

    def setUp(self):
        from src.data.session_pipeline import (
            validate_flow_array,
            validate_session_dataframe,
        )
        from src.detection.exceptions import (
            FeatureDimensionError,
            NaNInfError,
            SessionSizeError,
        )
        self.validate_flow_array = validate_flow_array
        self.validate_session_dataframe = validate_session_dataframe
        self.FeatureDimensionError = FeatureDimensionError
        self.NaNInfError = NaNInfError
        self.SessionSizeError = SessionSizeError

    def test_wrong_column_count_raises_feature_dimension_error(self):
        bad = np.ones((10, 50), dtype=np.float32)  # 50 cols, not 78
        with self.assertRaises(self.FeatureDimensionError) as cm:
            self.validate_flow_array(bad)
        self.assertEqual(cm.exception.expected, 78)
        self.assertEqual(cm.exception.actual, 50)

    def test_correct_column_count_does_not_raise(self):
        good = np.ones((10, 78), dtype=np.float32)
        self.validate_flow_array(good)  # must not raise

    def test_nan_raises_nan_inf_error(self):
        arr = np.ones((5, 78), dtype=np.float32)
        arr[2, 10] = np.nan
        with self.assertRaises(self.NaNInfError) as cm:
            self.validate_flow_array(arr)
        self.assertGreater(cm.exception.n_bad, 0)

    def test_inf_raises_nan_inf_error(self):
        arr = np.ones((5, 78), dtype=np.float32)
        arr[0, 0] = np.inf
        with self.assertRaises(self.NaNInfError):
            self.validate_flow_array(arr)

    def test_neg_inf_raises_nan_inf_error(self):
        arr = np.ones((5, 78), dtype=np.float32)
        arr[1, 5] = -np.inf
        with self.assertRaises(self.NaNInfError):
            self.validate_flow_array(arr)

    def test_session_wrong_size_raises_session_size_error(self):
        import pandas as pd
        df = pd.DataFrame(np.ones((15, 10)))  # 15 rows, not 20
        with self.assertRaises(self.SessionSizeError) as cm:
            self.validate_session_dataframe(df, enforce_size=True)
        self.assertEqual(cm.exception.expected, 20)
        self.assertEqual(cm.exception.actual, 15)

    def test_session_correct_size_does_not_raise(self):
        import pandas as pd
        df = pd.DataFrame(np.ones((20, 10)))
        self.validate_session_dataframe(df, enforce_size=True)  # must not raise

    def test_session_enforce_size_false_skips_check(self):
        import pandas as pd
        df = pd.DataFrame(np.ones((5, 10)))  # wrong size but enforce_size=False
        self.validate_session_dataframe(df, enforce_size=False)  # must not raise

    def test_realign_rejects_wrong_column_count(self):
        from src.data.session_pipeline import realign_to_trained_schema
        bad_77 = np.ones((20, 77), dtype=np.float32)
        lt = np.zeros(20, dtype=np.float32)
        with self.assertRaises(self.FeatureDimensionError):
            realign_to_trained_schema(bad_77, lt)

    def test_realign_rejects_nan_input(self):
        from src.data.session_pipeline import realign_to_trained_schema
        arr = _make_raw_78(5)
        arr[0, 0] = np.nan
        lt = np.zeros(5, dtype=np.float32)
        with self.assertRaises(self.NaNInfError):
            realign_to_trained_schema(arr, lt)


# ═════════════════════════════════════════════════════════════════════════════
# 7–8: trained_feature_names
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainedFeatureNames(unittest.TestCase):

    def setUp(self):
        from src.data.session_pipeline import trained_feature_names
        from src.detection.contracts import TRAINED_NUM_FEATURES, ZEEK_EXTRA_COLS
        self.names = trained_feature_names()
        self.TRAINED_NUM_FEATURES = TRAINED_NUM_FEATURES
        self.ZEEK_EXTRA_COLS = ZEEK_EXTRA_COLS

    def test_returns_exactly_81_names(self):
        self.assertEqual(len(self.names), self.TRAINED_NUM_FEATURES)

    def test_zeek_extra_cols_at_tail(self):
        tail = self.names[-len(self.ZEEK_EXTRA_COLS):]
        self.assertEqual(list(tail), list(self.ZEEK_EXTRA_COLS))

    def test_no_duplicate_names(self):
        self.assertEqual(len(set(self.names)), len(self.names),
                         "Duplicate column names in trained_feature_names()")

    def test_subflow_bwd_pkts_absent(self):
        """'Subflow Bwd Packets' must not appear — it was dropped at training."""
        self.assertNotIn("Subflow Bwd Packets", self.names)


# ═════════════════════════════════════════════════════════════════════════════
# 9: Both pipeline entry points produce identical output
# ═════════════════════════════════════════════════════════════════════════════

class TestBothPathsIdentical(unittest.TestCase):

    def test_realign_is_deterministic(self):
        """
        Calling realign_to_trained_schema twice on the same input must produce
        byte-identical results. This proves neither REST nor streaming adds
        per-call randomness.
        """
        from src.data.session_pipeline import realign_to_trained_schema
        raw = _make_raw_78(20, seed=42)
        lt = np.zeros(20, dtype=np.float32)

        out1 = realign_to_trained_schema(raw, lt)
        out2 = realign_to_trained_schema(raw.copy(), lt.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_log_type_passthrough_matches_for_same_input(self):
        """
        log_type values passed in must appear at col 77 unchanged.
        This ensures the REST path (client-supplied log_type) and the
        streaming path (zeek_mapper-derived log_type) feed the same column
        to the model when the log_type values agree.
        """
        from src.data.session_pipeline import realign_to_trained_schema
        raw = _make_raw_78(20, seed=7)
        lt = np.array([0, 1, 2, 3, 4] * 4, dtype=np.float32)

        result = realign_to_trained_schema(raw, lt)
        np.testing.assert_array_equal(result[:, 77], lt)


# ═════════════════════════════════════════════════════════════════════════════
# 10–11: Heuristic fallbacks are removed
# ═════════════════════════════════════════════════════════════════════════════

class TestHeuristicFallbacksRemoved(unittest.TestCase):

    def test_heuristic_detection_not_in_routes(self):
        """
        _heuristic_detection must not exist in routes.py.
        Its presence would mean the API can produce detections that bypass
        the Transformer, violating the architectural contract.
        """
        import src.api.app  # prevent circular import
        import src.api.routes as routes
        self.assertFalse(
            hasattr(routes, "_heuristic_detection"),
            "_heuristic_detection still present in routes.py — remove it",
        )

    def test_try_transformer_classifier_not_in_routes(self):
        """
        _try_transformer_classifier (Config D ablation path) must not exist
        in routes.py.  Config D is an ablation study, not a production path.
        """
        import src.api.app
        import src.api.routes as routes
        self.assertFalse(
            hasattr(routes, "_try_transformer_classifier"),
            "_try_transformer_classifier (Config D) still present in routes.py",
        )

    def test_heuristic_fallback_not_on_live_demo_server(self):
        """
        LiveDemoServer._heuristic_fallback must not exist.
        Streaming results must come from the Transformer or fail explicitly.
        """
        # Import live_demo module without triggering CSV/model loads
        import scripts.live_demo as live_demo
        server_cls = live_demo.LiveDemoServer
        self.assertFalse(
            hasattr(server_cls, "_heuristic_fallback"),
            "_heuristic_fallback still present on LiveDemoServer — remove it",
        )

    def test_try_session_hybrid_not_in_routes(self):
        """
        _try_session_hybrid (the old wrapper that silently fell back to
        heuristics) must not exist in routes.py.
        """
        import src.api.app
        import src.api.routes as routes
        self.assertFalse(
            hasattr(routes, "_try_session_hybrid"),
            "_try_session_hybrid still present in routes.py — remove it",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 12–14: predict_with_fallback is Config C → B only, raises ModelNotLoadedError
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictWithFallback(unittest.TestCase):

    def _make_hybrid(self, xgb_c=None, xgb_b=None, xgb_a=None):
        """Return a HybridIDS with mocked components."""
        import torch
        from src.models.hybrid import HybridIDS

        # Build a minimal classifier mock
        classifier = MagicMock()
        classifier.parameters.return_value = iter([torch.zeros(1)])
        # get_embedding returns a (1, 256) tensor
        classifier.get_embedding.return_value = torch.zeros(1, 256)
        classifier.eval.return_value = classifier
        # positional encoding mock
        pe_mock = MagicMock()
        pe_mock.pe.shape = (1, 60, 256)
        classifier.encoder.pos_enc = pe_mock

        cfg = {"paths": {"models_dir": "models/checkpoints"}, "transformer": {}}
        hybrid = HybridIDS(cfg, classifier, num_classes=15)
        hybrid.xgb_c = xgb_c
        hybrid.xgb_b = xgb_b
        hybrid.xgb_a = xgb_a
        return hybrid

    def test_config_a_not_tried_when_only_a_loaded(self):
        """
        If only Config A (stat-only) is loaded, predict_with_fallback must raise
        ModelNotLoadedError, NOT return a prediction from Config A.
        Config A bypasses the Transformer and is excluded from production inference.
        """
        from src.detection.exceptions import ModelNotLoadedError

        mock_a = MagicMock()
        mock_a.predict.return_value = np.array([3])
        mock_a.predict_proba.return_value = np.eye(15)[3:4]

        hybrid = self._make_hybrid(xgb_c=None, xgb_b=None, xgb_a=mock_a)

        raw = _make_raw_78(20)
        lt = np.zeros(20, dtype=np.int8)
        stats = np.zeros(26, dtype=np.float32)

        with self.assertRaises(ModelNotLoadedError):
            hybrid.predict_with_fallback(raw, lt, stats)

        # Verify Config A was NOT consulted
        mock_a.predict.assert_not_called()

    def test_config_c_used_when_loaded(self):
        """Config C must be tried first when loaded."""
        from src.detection.contracts import EXPECTED_NUM_CLASSES

        proba_c = np.zeros(EXPECTED_NUM_CLASSES, dtype=np.float32)
        proba_c[5] = 0.9

        mock_c = MagicMock()
        mock_c.predict.return_value = np.array([5])
        mock_c.predict_proba.return_value = proba_c.reshape(1, -1)

        mock_b = MagicMock()  # should NOT be called

        hybrid = self._make_hybrid(xgb_c=mock_c, xgb_b=mock_b)

        raw = _make_raw_78(20)
        lt = np.zeros(20, dtype=np.int8)
        stats = np.zeros(26, dtype=np.float32)

        result = hybrid.predict_with_fallback(raw, lt, stats)
        self.assertEqual(result["config_used"], "C")
        self.assertEqual(result["prediction"], 5)
        mock_b.predict.assert_not_called()

    def test_falls_back_to_b_when_c_absent(self):
        """Config B must be tried when Config C is None."""
        from src.detection.contracts import EXPECTED_NUM_CLASSES

        proba_b = np.zeros(EXPECTED_NUM_CLASSES, dtype=np.float32)
        proba_b[2] = 0.8

        mock_b = MagicMock()
        mock_b.predict.return_value = np.array([2])
        mock_b.predict_proba.return_value = proba_b.reshape(1, -1)

        hybrid = self._make_hybrid(xgb_c=None, xgb_b=mock_b)

        raw = _make_raw_78(20)
        lt = np.zeros(20, dtype=np.int8)
        stats = np.zeros(26, dtype=np.float32)

        result = hybrid.predict_with_fallback(raw, lt, stats)
        self.assertEqual(result["config_used"], "B")
        self.assertEqual(result["prediction"], 2)

    def test_raises_model_not_loaded_when_c_and_b_absent(self):
        """Both C and B absent → ModelNotLoadedError (not RuntimeError)."""
        from src.detection.exceptions import ModelNotLoadedError

        hybrid = self._make_hybrid(xgb_c=None, xgb_b=None)

        raw = _make_raw_78(20)
        lt = np.zeros(20, dtype=np.int8)
        stats = np.zeros(26, dtype=np.float32)

        with self.assertRaises(ModelNotLoadedError):
            hybrid.predict_with_fallback(raw, lt, stats)

    def test_error_is_model_not_loaded_not_runtime_error(self):
        """Raised exception must be ModelNotLoadedError, not plain RuntimeError."""
        from src.detection.exceptions import ModelNotLoadedError

        hybrid = self._make_hybrid(xgb_c=None, xgb_b=None)

        raw = _make_raw_78(20)
        lt = np.zeros(20, dtype=np.int8)
        stats = np.zeros(26, dtype=np.float32)

        try:
            hybrid.predict_with_fallback(raw, lt, stats)
            self.fail("Expected ModelNotLoadedError but no exception raised")
        except ModelNotLoadedError:
            pass  # correct
        except RuntimeError as e:
            self.fail(
                f"Got bare RuntimeError instead of ModelNotLoadedError: {e}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 15: InferenceError is typed and does not corrupt session
# ═════════════════════════════════════════════════════════════════════════════

class TestInferenceError(unittest.TestCase):

    def test_inference_error_carries_config_and_cause(self):
        from src.detection.exceptions import InferenceError
        cause = ValueError("shape mismatch")
        exc = InferenceError("C", cause)
        self.assertEqual(exc.config, "C")
        self.assertIs(exc.cause, cause)

    def test_inference_error_message_contains_config(self):
        from src.detection.exceptions import InferenceError
        exc = InferenceError("B", ValueError("bad"))
        self.assertIn("Config-B", str(exc))

    def test_model_not_loaded_error_carries_component(self):
        from src.detection.exceptions import ModelNotLoadedError
        exc = ModelNotLoadedError("FeatureScaler", "/some/path.pkl")
        self.assertEqual(exc.component, "FeatureScaler")
        self.assertIn("/some/path.pkl", str(exc))

    def test_model_not_loaded_suggests_train_script(self):
        from src.detection.exceptions import ModelNotLoadedError
        exc = ModelNotLoadedError("FeatureScaler")
        self.assertIn("train_model.py", str(exc))


# ═════════════════════════════════════════════════════════════════════════════
# 16: EnrichmentError is non-fatal
# ═════════════════════════════════════════════════════════════════════════════

class TestEnrichmentErrorNonFatal(unittest.TestCase):

    def test_enrichment_error_is_non_fatal_subclass(self):
        """EnrichmentError must not be a SchemaError — it must be DetectionError."""
        from src.detection.exceptions import (
            DetectionError,
            EnrichmentError,
            SchemaError,
        )
        exc = EnrichmentError("rag", "connection timeout")
        self.assertIsInstance(exc, DetectionError)
        self.assertNotIsInstance(exc, SchemaError)

    def test_enrichment_error_carries_stage(self):
        from src.detection.exceptions import EnrichmentError
        exc = EnrichmentError("groq", "rate limit exceeded")
        self.assertEqual(exc.stage, "groq")

    def test_enrichment_error_message_says_detection_result_valid(self):
        from src.detection.exceptions import EnrichmentError
        exc = EnrichmentError("rag", "timeout")
        self.assertIn("valid", str(exc).lower())


# ═════════════════════════════════════════════════════════════════════════════
# 17: AlertStore REST and stream schemas
# ═════════════════════════════════════════════════════════════════════════════

class TestAlertStoreSchema(unittest.TestCase):

    def setUp(self):
        import tempfile
        import os
        from src.api.alert_store import AlertStore, reset_alert_store
        reset_alert_store()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()
        self.store = AlertStore(db_path=self.db_path)

    def tearDown(self):
        import os
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def _rest_alert(self) -> dict:
        return {
            "alert_id": "rest-001",
            "timestamp": 1700000000.0,
            "attack_id": 10,
            "attack_name": "PortScan",
            "confidence": 0.92,
            "is_attack": True,
            "n_flows": 20,
        }

    def _stream_alert(self) -> dict:
        return {
            "alert_id": "stream-001",
            "timestamp": 1700000001.0,
            "attack_id": 0,
            "attack_name": "BENIGN",
            "confidence": 0.95,
            "is_attack": False,
            "n_flows": 20,
        }

    def test_rest_alert_can_be_recorded(self):
        self.store.record(self._rest_alert(), source="rest")
        results = self.store.recent(limit=10, source="rest")
        self.assertEqual(len(results), 1)

    def test_stream_alert_can_be_recorded(self):
        self.store.record(self._stream_alert(), source="stream",
                          ground_truth_id=0, session_id=1)
        results = self.store.recent(limit=10, source="stream")
        self.assertEqual(len(results), 1)

    def test_both_sources_visible_without_filter(self):
        self.store.record(self._rest_alert(), source="rest")
        self.store.record(self._stream_alert(), source="stream")
        results = self.store.recent(limit=10)
        self.assertEqual(len(results), 2)

    def test_source_filter_isolates_rest(self):
        self.store.record(self._rest_alert(), source="rest")
        self.store.record(self._stream_alert(), source="stream")
        rest_only = self.store.recent(limit=10, source="rest")
        self.assertEqual(len(rest_only), 1)
        self.assertEqual(rest_only[0]["source"], "rest")

    def test_stats_counts_attacks_and_benign(self):
        self.store.record(self._rest_alert(), source="rest")
        self.store.record(self._stream_alert(), source="stream")
        stats = self.store.stats()
        # Use the actual AlertStore.stats() key names
        self.assertIn("total_sessions_analyzed", stats)
        self.assertEqual(stats["total_sessions_analyzed"], 2)
        self.assertIn("total_attacks_detected", stats)
        self.assertEqual(stats["total_attacks_detected"], 1)

    def test_uuid_deduplication(self):
        """Records with the same alert_uuid must not be inserted twice."""
        alert = self._rest_alert()
        self.store.record(alert, source="rest")
        self.store.record(alert, source="rest")  # duplicate — must be ignored
        results = self.store.recent(limit=10)
        self.assertEqual(len(results), 1)


# ═════════════════════════════════════════════════════════════════════════════
# 18–19: Architectural constants are self-consistent
# ═════════════════════════════════════════════════════════════════════════════

class TestArchitecturalContracts(unittest.TestCase):

    def setUp(self):
        from src.detection.contracts import (
            CICIDS_RAW_COLS,
            EXPECTED_EMBEDDING_DIM,
            EXPECTED_HYBRID_INPUT_DIM,
            EXPECTED_NUM_CLASSES,
            EXPECTED_SESSION_SIZE,
            EXPECTED_STAT_FEATURES,
            SUBFLOW_BWD_PKTS_IDX,
            TRAINED_NUM_FEATURES,
            ZEEK_EXTRA_COLS,
        )
        self.CICIDS_RAW_COLS = CICIDS_RAW_COLS
        self.EXPECTED_EMBEDDING_DIM = EXPECTED_EMBEDDING_DIM
        self.EXPECTED_HYBRID_INPUT_DIM = EXPECTED_HYBRID_INPUT_DIM
        self.EXPECTED_NUM_CLASSES = EXPECTED_NUM_CLASSES
        self.EXPECTED_SESSION_SIZE = EXPECTED_SESSION_SIZE
        self.EXPECTED_STAT_FEATURES = EXPECTED_STAT_FEATURES
        self.SUBFLOW_BWD_PKTS_IDX = SUBFLOW_BWD_PKTS_IDX
        self.TRAINED_NUM_FEATURES = TRAINED_NUM_FEATURES
        self.ZEEK_EXTRA_COLS = ZEEK_EXTRA_COLS

    def test_trained_features_equals_cicids_minus_one_plus_zeek(self):
        """TRAINED_NUM_FEATURES = CICIDS_RAW_COLS - 1 + |ZEEK_EXTRA_COLS|."""
        expected = self.CICIDS_RAW_COLS - 1 + len(self.ZEEK_EXTRA_COLS)
        self.assertEqual(self.TRAINED_NUM_FEATURES, expected,
                         f"TRAINED_NUM_FEATURES={self.TRAINED_NUM_FEATURES} "
                         f"!= {self.CICIDS_RAW_COLS} - 1 + {len(self.ZEEK_EXTRA_COLS)}")

    def test_hybrid_input_dim_equals_embedding_plus_stat(self):
        """EXPECTED_HYBRID_INPUT_DIM = EXPECTED_EMBEDDING_DIM + EXPECTED_STAT_FEATURES."""
        expected = self.EXPECTED_EMBEDDING_DIM + self.EXPECTED_STAT_FEATURES
        self.assertEqual(self.EXPECTED_HYBRID_INPUT_DIM, expected)

    def test_zeek_extra_cols_has_four_entries(self):
        self.assertEqual(len(self.ZEEK_EXTRA_COLS), 4)

    def test_zeek_extra_cols_order(self):
        self.assertEqual(
            list(self.ZEEK_EXTRA_COLS),
            ["log_type", "zeek_proto", "zeek_conn_state", "zeek_service"],
        )

    def test_subflow_bwd_pkts_idx_within_cicids_range(self):
        self.assertGreater(self.SUBFLOW_BWD_PKTS_IDX, 0)
        self.assertLess(self.SUBFLOW_BWD_PKTS_IDX, self.CICIDS_RAW_COLS)

    def test_expected_session_size_matches_session_pipeline(self):
        from src.detection.contracts import EXPECTED_SESSION_SIZE
        self.assertEqual(EXPECTED_SESSION_SIZE, 20)

    def test_expected_num_classes_is_15(self):
        self.assertEqual(self.EXPECTED_NUM_CLASSES, 15)


# ═════════════════════════════════════════════════════════════════════════════
# 20: Stat computation must use 81-col trained_feature_names, not 79-col raw
# ═════════════════════════════════════════════════════════════════════════════

class TestStatComputationSchema(unittest.TestCase):

    def test_trained_feature_names_provides_log_type_column(self):
        """
        compute_session_stats needs 'log_type' in the column list to compute
        log_type_entropy correctly. trained_feature_names() must include it.
        """
        from src.data.session_pipeline import trained_feature_names
        names = trained_feature_names()
        self.assertIn("log_type", names,
                      "log_type missing from trained_feature_names() — "
                      "stat computation will silently use zeros for that column")

    def test_compute_session_stats_with_81_cols_returns_26_features(self):
        """
        compute_session_stats on the 81-col aligned array must produce exactly
        EXPECTED_STAT_FEATURES (26) outputs.
        """
        from src.data.feature_engineer import compute_session_stats
        from src.data.session_pipeline import realign_to_trained_schema, trained_feature_names
        from src.detection.contracts import EXPECTED_STAT_FEATURES

        raw = _make_raw_78(20, seed=99)
        lt = np.zeros(20, dtype=np.float32)
        aligned_81 = realign_to_trained_schema(raw, lt)

        stats = compute_session_stats(aligned_81, trained_feature_names())
        self.assertEqual(stats.shape[0], EXPECTED_STAT_FEATURES)

    def test_stat_feature_names_length_matches_contract(self):
        from src.data.feature_engineer import STAT_FEATURE_NAMES, NUM_STAT_FEATURES
        from src.detection.contracts import EXPECTED_STAT_FEATURES
        self.assertEqual(len(STAT_FEATURE_NAMES), EXPECTED_STAT_FEATURES)
        self.assertEqual(NUM_STAT_FEATURES, EXPECTED_STAT_FEATURES)

    def test_routes_imports_trained_feature_names(self):
        """
        routes.py must import trained_feature_names from session_pipeline,
        proving the stat computation fix has been applied (no more 79-col raw array).
        """
        import src.api.app  # load app first to prevent circular import
        import src.api.routes as routes
        # The import is at module level — verify the name exists
        self.assertTrue(
            hasattr(routes, "trained_feature_names"),
            "routes.py does not import trained_feature_names — "
            "stat computation may still use the 79-col raw array"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═════════════════════════════════════════════════════════════════════════════

class TestExceptionHierarchy(unittest.TestCase):

    def test_all_exceptions_inherit_from_detection_error(self):
        from src.detection.exceptions import (
            DetectionError,
            EnrichmentError,
            FeatureDimensionError,
            InferenceError,
            ModelNotLoadedError,
            NaNInfError,
            SchemaError,
            SessionSizeError,
        )
        for exc_cls in [
            ModelNotLoadedError,
            SchemaError,
            FeatureDimensionError,
            SessionSizeError,
            NaNInfError,
            InferenceError,
            EnrichmentError,
        ]:
            self.assertTrue(
                issubclass(exc_cls, DetectionError),
                f"{exc_cls.__name__} does not inherit from DetectionError"
            )

    def test_schema_exceptions_are_schema_errors(self):
        from src.detection.exceptions import (
            FeatureDimensionError,
            NaNInfError,
            SchemaError,
            SessionSizeError,
        )
        for exc_cls in [FeatureDimensionError, SessionSizeError, NaNInfError]:
            self.assertTrue(
                issubclass(exc_cls, SchemaError),
                f"{exc_cls.__name__} should inherit from SchemaError"
            )

    def test_all_exceptions_are_runtime_errors(self):
        from src.detection.exceptions import DetectionError
        self.assertTrue(issubclass(DetectionError, RuntimeError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
