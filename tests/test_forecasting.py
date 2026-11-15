import unittest

import numpy as np
import pandas as pd

from forecasting import forecast_series, fit_gpr, rolling_origin_evaluation


class ForecastTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-11-01", periods=11)
        values = 1_000 + 180 * np.cos(2 * np.pi * np.arange(11) / 7)
        self.series = pd.Series(values, index=index)

    def test_gpr_fit_uses_fixed_weekly_period(self):
        fit = fit_gpr(self.series)
        self.assertIn("periodicity=7", str(fit.model.kernel_))

    def test_rolling_origin_evaluation(self):
        metrics, comparison = rolling_origin_evaluation(self.series)
        self.assertEqual(set(metrics["Model"]), {"GPR", "Seasonal Naive"})
        self.assertEqual(len(comparison), 4)
        self.assertTrue(np.isfinite(metrics[["MAE", "RMSE", "MAPE (%)"]]).all().all())

    def test_three_day_forecast_has_intervals(self):
        future = forecast_series(self.series, 3)
        self.assertEqual(len(future), 3)
        self.assertTrue((future["Lower 95%"] <= future["Predicted"]).all())
        self.assertTrue((future["Predicted"] <= future["Upper 95%"]).all())


if __name__ == "__main__":
    unittest.main()

