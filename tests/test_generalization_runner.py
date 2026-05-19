import unittest

from histoomnist.eval.generalization_runner import compact_task_dir_name


class GeneralizationRunnerTest(unittest.TestCase):
    def test_compact_task_dir_name_keeps_short_slugs(self):
        self.assertEqual(compact_task_dir_name("breast"), "breast")

    def test_compact_task_dir_name_hashes_long_slugs(self):
        slug = "spatial_multimodal_analysis_maldi_msi_and_spatial_transcriptomics_within_the_same_tissue_section"

        compact = compact_task_dir_name(slug, max_length=40)

        self.assertLessEqual(len(compact), 40)
        self.assertTrue(compact.startswith("spatial_multimodal_analysis"))
        self.assertNotEqual(compact, slug)


if __name__ == "__main__":
    unittest.main()
