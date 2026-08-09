import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from main import LUT, LUTUpdateStats, record_attention_update_rows


def make_lut(n_tables, n_rows):
    lut = LUT(n_tables)
    lut.S = np.zeros((n_tables, n_rows, 1), dtype=np.float32)
    lut.y_dim = 1
    return lut


class LUTUpdateStatsTest(unittest.TestCase):
    def test_record_simple_tracks_row_counts_and_j_counts(self):
        stats = LUTUpdateStats()
        lut = make_lut(2, 4)
        j = np.array([[0, 1], [1, 3]], dtype=np.int32)

        stats.record_simple("simple", lut, j)

        expected = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=np.uint64)
        np.testing.assert_array_equal(stats.counts["simple"], expected)
        np.testing.assert_array_equal(stats.j_counts["simple"], expected)

    def test_record_attention_rows_tracks_concatenated_components(self):
        args = SimpleNamespace(
            context_size=3,
            n_t=2,
            n_c=2,
            positional_dim=1,
            shift_qk=3,
        )
        stats = LUTUpdateStats()
        lut = make_lut(2, 1 << (2 * args.n_c + args.positional_dim))
        j_v = np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int32)
        j_pe = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)

        record_attention_update_rows(stats, "head_0_0_V", lut, j_v, j_pe, args)

        expected_rows = np.zeros_like(stats.counts["head_0_0_V"])
        expected_rows[0, [17, 8, 13]] = 1
        expected_rows[1, [26, 3, 6]] = 1
        np.testing.assert_array_equal(stats.counts["head_0_0_V"], expected_rows)

        expected_q = np.array([[0, 2, 1, 0], [2, 0, 0, 1]], dtype=np.uint64)
        expected_k = np.array([[2, 0, 1, 0], [0, 2, 0, 1]], dtype=np.uint64)
        expected_pe = np.array([[1, 2], [2, 1]], dtype=np.uint64)
        np.testing.assert_array_equal(stats.j_component_counts[("head_0_0_V", "q")], expected_q)
        np.testing.assert_array_equal(stats.j_component_counts[("head_0_0_V", "k")], expected_k)
        np.testing.assert_array_equal(stats.j_component_counts[("head_0_0_V", "pe")], expected_pe)

    def test_record_new_attention_final_tracks_components_and_joint(self):
        args = SimpleNamespace(context_size=3, n_t=2, n_c=2, positional_dim=1)
        stats = LUTUpdateStats()
        lut = make_lut(2, 1 << (args.n_c + args.positional_dim))
        final_v_j = np.array([[0, 1], [2, 3], [1, 0]], dtype=np.int32)
        qkpe_j = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32)

        stats.record_new_attention_final("head_0_0_Final", lut, final_v_j, qkpe_j, args)

        expected_rows = np.zeros_like(stats.counts["head_0_0_Final"])
        expected_rows[0, [1, 0, 5]] = 1
        expected_rows[1, [2, 3, 7]] = 1
        np.testing.assert_array_equal(stats.counts["head_0_0_Final"], expected_rows)

        expected_qkpe = np.array([[1, 2], [1, 2]], dtype=np.uint64)
        expected_final_v = np.array([[2, 0, 1, 0], [0, 2, 0, 1]], dtype=np.uint64)
        np.testing.assert_array_equal(
            stats.j_component_counts[("head_0_0_Final", "qkpe")], expected_qkpe)
        np.testing.assert_array_equal(
            stats.j_component_counts[("head_0_0_Final", "final_v")], expected_final_v)

        joint = stats.j_joint_counts[("head_0_0_Final", "qkpe", "final_v")]
        expected_joint = np.zeros((2, 2, 4), dtype=np.uint64)
        expected_joint[0, 1, 0] = 1
        expected_joint[0, 0, 0] = 1
        expected_joint[0, 1, 2] = 1
        expected_joint[1, 0, 1] = 1
        expected_joint[1, 1, 1] = 1
        expected_joint[1, 1, 3] = 1
        np.testing.assert_array_equal(joint, expected_joint)

    def test_record_new_attention_final_shared_trees_aggregate_physical_rows(self):
        args = SimpleNamespace(context_size=3, n_t=4, n_c=1, positional_dim=1)
        stats = LUTUpdateStats()
        lut = make_lut(2, 1 << (args.n_c + args.positional_dim))
        lut.trees = np.array([0, 0, 1, 1], dtype=np.int32)
        final_v_j = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int32)
        qkpe_j = np.array([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.int32)
        y_grad = np.array([[0.0], [2.0], [3.0]], dtype=np.float32)

        stats.record_new_attention_final("shared_Final", lut, final_v_j, qkpe_j, args)
        stats.record_new_attention_final_coords("shared_Final", lut, y_grad, args)

        expected_rows = np.zeros_like(stats.counts["shared_Final"])
        for tok_i in range(1, args.context_size):
            for tok_j in range(tok_i):
                pair = tok_i * (tok_i - 1) // 2 + tok_j
                for t, tree in enumerate(lut.trees):
                    row = (final_v_j[tok_j, t] << args.positional_dim) | qkpe_j[pair, t]
                    expected_rows[tree, row] += 1
        np.testing.assert_array_equal(stats.counts["shared_Final"], expected_rows)
        np.testing.assert_array_equal(
            stats.coord_counts["shared_Final"],
            np.array([[6], [6]], dtype=np.uint64))

    def test_save_writes_j_stats_sidecar_without_changing_row_npz_keys(self):
        stats = LUTUpdateStats()
        lut = make_lut(1, 4)
        stats.record_simple("output", lut, np.array([[2]], dtype=np.int32))

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "lut_update_stats.npz"
            (summary_path, hist_path, coord_path, coord_summary_path,
             j_path, j_summary_path, j_top_path) = stats.save(str(raw_path))
            self.assertTrue(raw_path.exists())
            self.assertTrue(Path(summary_path).exists())
            self.assertTrue(Path(hist_path).exists())
            self.assertTrue(Path(coord_path).exists())
            self.assertTrue(Path(coord_summary_path).exists())
            self.assertTrue(Path(j_path).exists())
            self.assertTrue(Path(j_summary_path).exists())
            self.assertTrue(Path(j_top_path).exists())

            with np.load(raw_path) as data:
                self.assertEqual(data.files, ["output"])
            with np.load(j_path) as data:
                self.assertIn("simple__output", data.files)


if __name__ == "__main__":
    unittest.main()
