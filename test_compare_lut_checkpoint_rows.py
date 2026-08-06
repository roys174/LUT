import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compare_lut_checkpoint_rows as compare


def write_checkpoint(path, ffn=None, output=None, head=None):
    arrays = {}
    if ffn is not None:
        arrays["ffn_0_S"] = np.asarray(ffn, dtype=np.float32)
    if output is not None:
        arrays["output_S"] = np.asarray(output, dtype=np.float32)
    if head is not None:
        arrays["head_0_1_V_S"] = np.asarray(head, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class CheckpointRowDiffTest(unittest.TestCase):
    def test_explicit_paths_compare_first_to_last_and_count_changed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = np.zeros((2, 4, 3), dtype=np.float32)
            mid = base.copy()
            final = base.copy()
            final[0, 1, 0] = 1.0
            final[0, 1, 2] = 2.0
            final[1, 3, 1] = 3.0
            write_checkpoint(tmp / "model_step_0.npz", ffn=base)
            write_checkpoint(tmp / "model_step_5.npz", ffn=mid)
            write_checkpoint(tmp / "model.npz", ffn=final)
            out_dir = tmp / "out"

            code = compare.main_with_args([
                str(tmp / "model_step_0.npz"),
                str(tmp / "model_step_5.npz"),
                str(tmp / "model.npz"),
                "--out-dir", str(out_dir),
                "--no-plots",
            ])

            self.assertEqual(code, 0)
            rows = read_csv(out_dir / "checkpoint_row_diffs.csv")
            by_table = {int(row["table"]): row for row in rows}
            self.assertEqual(int(by_table[0]["changed_rows"]), 1)
            self.assertEqual(int(by_table[0]["changed_elements"]), 2)
            self.assertEqual(int(by_table[1]["changed_rows"]), 1)
            self.assertEqual(int(by_table[1]["changed_elements"]), 1)

    def test_directory_input_ignores_non_model_npz_and_sorts_model_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            arr = np.zeros((1, 2, 1), dtype=np.float32)
            write_checkpoint(tmp / "model_step_10.npz", ffn=arr + 10)
            write_checkpoint(tmp / "model_step_0.npz", ffn=arr)
            write_checkpoint(tmp / "model.npz", ffn=arr + 20)
            np.savez_compressed(tmp / "lut_update_stats_step_5.npz", ffn_0=np.ones((1, 2)))

            checkpoints = compare.resolve_checkpoint_paths([str(tmp)])

            self.assertEqual([ckpt.path.name for ckpt in checkpoints], [
                "model_step_0.npz",
                "model_step_10.npz",
                "model.npz",
            ])

    def test_tolerance_can_suppress_small_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = np.zeros((1, 3, 2), dtype=np.float32)
            final = base.copy()
            final[0, 0, 0] = 0.001
            final[0, 2, 1] = 0.5
            write_checkpoint(tmp / "model_step_0.npz", ffn=base)
            write_checkpoint(tmp / "model.npz", ffn=final)
            out_dir = tmp / "out"

            code = compare.main_with_args([
                str(tmp), "--out-dir", str(out_dir), "--no-plots", "--atol", "0.01",
            ])

            self.assertEqual(code, 0)
            rows = read_csv(out_dir / "checkpoint_row_diffs.csv")
            self.assertEqual(int(rows[0]["changed_rows"]), 1)
            self.assertEqual(int(rows[0]["changed_elements"]), 1)

    def test_output_lut_is_included_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = np.zeros((1, 2, 1), dtype=np.float32)
            final = base.copy()
            final[0, 1, 0] = 1.0
            write_checkpoint(tmp / "model_step_0.npz", ffn=base, output=base)
            write_checkpoint(tmp / "model.npz", ffn=base, output=final)
            out_dir = tmp / "out"

            code = compare.main_with_args([str(tmp), "--out-dir", str(out_dir), "--no-plots"])

            self.assertEqual(code, 0)
            rows = read_csv(out_dir / "checkpoint_row_diffs.csv")
            self.assertIn("output", {row["lut"] for row in rows})

    def test_mismatch_fails_by_default_and_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_checkpoint(
                tmp / "model_step_0.npz",
                ffn=np.zeros((1, 2, 1)),
                output=np.zeros((1, 2, 1)),
            )
            write_checkpoint(
                tmp / "model.npz",
                ffn=np.zeros((1, 3, 1)),
                output=np.zeros((1, 2, 1)),
            )

            self.assertEqual(compare.main_with_args([str(tmp), "--out-dir", str(tmp / "fail"), "--no-plots"]), 1)
            self.assertEqual(compare.main_with_args([
                str(tmp), "--out-dir", str(tmp / "ok"), "--no-plots", "--allow-mismatch",
            ]), 0)
            rows = read_csv(tmp / "ok" / "checkpoint_row_diffs.csv")
            self.assertEqual({row["lut"] for row in rows}, {"output"})

    def test_plots_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = np.zeros((1, 4, 1), dtype=np.float32)
            final = base.copy()
            final[0, 2, 0] = 1.0
            write_checkpoint(tmp / "model_step_0.npz", ffn=base)
            write_checkpoint(tmp / "model.npz", ffn=final)
            out_dir = tmp / "out"

            with patch.dict("os.environ", {"MPLBACKEND": "Agg"}):
                code = compare.main_with_args([str(tmp), "--out-dir", str(out_dir)])

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "ffn_0_row-l2.png").exists())
            self.assertFalse((out_dir / "ffn_0_binary.png").exists())

    def test_plot_metric_values_are_computed_from_known_row_deltas(self):
        left = np.zeros((3, 3), dtype=np.float32)
        right = np.array([
            [3.0, 4.0, 0.0],
            [1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        l2_values, l2_metrics = compare.compare_table(left, right, 0.0, 0.0, "row-l2")
        rmse_values, _ = compare.compare_table(left, right, 0.0, 0.0, "row-rmse")
        lmin_values, lmin_metrics = compare.compare_table(left, right, 0.0, 0.0, "row-lmin")

        np.testing.assert_allclose(l2_values, [5.0, 3.0, 0.0])
        np.testing.assert_allclose(rmse_values, [5.0 / np.sqrt(3.0), np.sqrt(3.0), 0.0])
        np.testing.assert_allclose(lmin_values, [0.0, 1.0, 0.0])
        self.assertEqual(l2_metrics["plot_metric_nonzero_rows"], 2)
        self.assertAlmostEqual(l2_metrics["plot_metric_nonzero_pct"], 100.0 * 2.0 / 3.0)
        self.assertEqual(lmin_metrics["plot_metric_nonzero_rows"], 1)
        self.assertAlmostEqual(lmin_metrics["plot_metric_nonzero_pct"], 100.0 / 3.0)

    def test_plot_metric_uses_tolerance_adjusted_deltas_for_nonzero_percentage(self):
        left = np.zeros((2, 2), dtype=np.float32)
        right = np.array([[0.001, 0.5], [0.001, 0.002]], dtype=np.float32)

        values, metrics = compare.compare_table(left, right, 0.01, 0.0, "row-lmin")

        np.testing.assert_allclose(values, [0.0, 0.0])
        self.assertEqual(metrics["changed_rows"], 1)
        self.assertEqual(metrics["plot_metric_nonzero_rows"], 0)

    def test_binary_plot_metric_preserves_old_indicator_values(self):
        left = np.zeros((2, 2), dtype=np.float32)
        right = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        values, metrics = compare.compare_table(left, right, 0.0, 0.0, "binary")

        np.testing.assert_allclose(values, [0.0, 1.0])
        self.assertEqual(metrics["plot_metric_nonzero_rows"], 1)

    def test_selected_plot_metric_controls_png_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = np.zeros((1, 4, 2), dtype=np.float32)
            final = base.copy()
            final[0, 2, :] = [1.0, 2.0]
            write_checkpoint(tmp / "model_step_0.npz", ffn=base)
            write_checkpoint(tmp / "model.npz", ffn=final)
            out_dir = tmp / "out"

            with patch.dict("os.environ", {"MPLBACKEND": "Agg"}):
                code = compare.main_with_args([
                    str(tmp), "--out-dir", str(out_dir), "--plot-metric", "row-lmin",
                ])

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "ffn_0_row-lmin.png").exists())
            self.assertFalse((out_dir / "ffn_0_binary.png").exists())
            rows = read_csv(out_dir / "checkpoint_row_diffs.csv")
            self.assertEqual(rows[0]["plot_metric"], "row-lmin")
            self.assertEqual(int(rows[0]["plot_metric_nonzero_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
