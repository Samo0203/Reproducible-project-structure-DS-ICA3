
import unittest

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    RAW_DATA_PATH,
    FIGURES_DIR,
    REPORTS_DIR
)


class TestProjectConfig(unittest.TestCase):

    def test_random_state(self):
        self.assertEqual(RANDOM_STATE, 42)

    def test_test_size(self):
        self.assertEqual(TEST_SIZE, 0.20)

    def test_cv_folds(self):
        self.assertEqual(CV_FOLDS, 5)

    def test_raw_data_path(self):
        self.assertEqual(RAW_DATA_PATH.name, "dataset.csv")

    def test_output_directories(self):
        self.assertEqual(FIGURES_DIR.name, "figures")
        self.assertEqual(REPORTS_DIR.name, "reports")


if __name__ == "__main__":
    unittest.main()
