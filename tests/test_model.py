import unittest

import pandas as pd

from model import FEATURE_COLUMNS, TARGET_COLUMN, temporal_split, train_model


class ModelTests(unittest.TestCase):
    def test_temporal_split_keeps_each_date_in_one_partition(self):
        dataframe = pd.DataFrame(
            {
                "tanggal": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            }
        )

        train, test = temporal_split(dataframe, test_fraction=0.34)

        self.assertTrue(set(train["tanggal"]).isdisjoint(set(test["tanggal"])))
        self.assertLess(train["tanggal"].max(), test["tanggal"].min())

    def test_model_uses_expected_predictors_without_target_derived_columns(self):
        self.assertIn(TARGET_COLUMN, ("hasil_kg",))
        self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)
        self.assertNotIn("hasil_log", FEATURE_COLUMNS)
        self.assertNotIn("hasil_per_jam", FEATURE_COLUMNS)
        self.assertNotIn("layak_melaut", FEATURE_COLUMNS)

    def test_training_evaluates_on_later_dates(self):
        dataframe = pd.read_csv("dataset_final.csv")
        result = train_model(dataframe)

        self.assertGreater(result.train_rows, 0)
        self.assertGreater(result.test_rows, 0)
        self.assertGreater(result.metrics["mae"], 0)
        self.assertTrue(pd.notna(result.metrics["rmse"]))
        self.assertTrue(pd.notna(result.metrics["r2"]))
        self.assertGreater(len(result.feature_importance), 0)


if __name__ == "__main__":
    unittest.main()
