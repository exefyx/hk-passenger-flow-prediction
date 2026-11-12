import unittest
import numpy as np
import pandas as pd
from forecasting import make_windows, same_weekday_forecast, evaluate_models, predict_future

class ForecastTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range('2025-01-01', periods=220)
        vals = 1000 + 190 * np.sin(2*np.pi*idx.dayofweek.to_numpy()/7) + np.arange(220)*.12
        self.series = pd.Series(vals, index=idx)

    def test_windows(self):
        x, y = make_windows(np.arange(40), 28)
        self.assertEqual(x.shape, (12,28,1))
        self.assertEqual(y.shape, (12,1))

    def test_seasonal(self):
        dates = pd.date_range(self.series.index[-1]+pd.Timedelta(days=1), periods=3)
        preds = same_weekday_forecast(self.series, dates)
        self.assertEqual(len(preds),3)
        self.assertTrue(np.all(preds > 0))

    def test_comparison_and_future(self):
        summary, actual_vs_pred, _ = evaluate_models(self.series, test_days=30)
        self.assertEqual(set(summary['Model']), {'Seasonal Naive','Random Forest','LSTM'})
        self.assertEqual(len(actual_vs_pred),30)
        self.assertTrue(np.isfinite(summary[['MAE','RMSE']].to_numpy()).all())
        future = predict_future(self.series,3,'LSTM')
        self.assertEqual(len(future), 3)
        self.assertTrue((future['Predicted trips']>=0).all())

if __name__ == '__main__':
    unittest.main()
