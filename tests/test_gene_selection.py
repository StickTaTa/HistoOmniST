import tempfile
import unittest
from pathlib import Path

from histoomnist.data.gene_selection import selected_genes_from_config


class GeneSelectionTest(unittest.TestCase):
    def test_selected_gene_names_resolve_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            genes_path = tmp_path / "data" / "genes.txt"
            genes_path.parent.mkdir(parents=True)
            genes_path.write_text("G1\nG2\n", encoding="utf-8")

            task_dir = tmp_path / "results" / "nested" / "task"
            task_dir.mkdir(parents=True)
            relative_path = Path("..") / ".." / ".." / "data" / "genes.txt"
            cfg = {"data": {"gene_names_path": str(relative_path)}}

            gene_names, gene_indices = selected_genes_from_config(cfg, base_dir=task_dir)

            self.assertEqual(gene_names, ["G1", "G2"])
            self.assertIsNone(gene_indices)


if __name__ == "__main__":
    unittest.main()
