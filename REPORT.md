# Real Time Analytics of Passenger Flow and Train Headway

## Abstract

This report studies a staged approach to passenger-flow prediction and metro congestion management. The current prototype aggregates official Hong Kong Immigration Department records for 1-11 November 2026 into daily arrival, departure and total clearance trips. Two Gaussian Process Regression models with a fixed seven-day periodic kernel forecast the following three days and return prediction intervals. The public dataset is suitable for demonstrating data ingestion, probabilistic forecasting and time-ordered validation, but it does not contain MTR station occupancy or train headway. The dissertation therefore separates the demonstrated public-data model from later station-level headway prediction and congestion-control simulation.

**Keywords:** passenger flow, Gaussian Process Regression, train headway, urban rail transit, congestion control

## 1 Introduction

The project objective is to develop real-time analytics that predict passenger flow and train headway, identify potential congestion, and evaluate mitigation strategies in simulation. The current public-data stage answers a narrower question: whether eleven daily observations can support a transparent three-day machine-learning forecast. This boundary prevents immigration-clearance trips from being presented as MTR platform crowding.

The report follows the evidence chain used in passenger-flow studies: problem definition, theoretical basis, data and preprocessing, model design, validation, results, limitations and operational extension.

## 2 Literature Review

The papers were assessed on five dimensions: fit with the dissertation, recency, publication or code license, technical stack, and deployment difficulty. Scores use a five-point scale, where five means a stronger fit, a more recent source, a more reusable license, or easier deployment.

| Rank | Paper and latest version | Fit | Recency | License and reproducibility | Stack fit | Deployment | Total | Use in this project |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Song, Feng and Li, *Urban Rail Transit Passenger Flow Prediction with Gaussian Process Regression Based on CEEMDAN*, 11 July 2025 | 5 | 4 | 1 | 5 | 3 | 18 | Supports GPR as the immediate passenger-flow model; decomposition and clustering are omitted because eleven samples are insufficient |
| 2 | Usama and Koutsopoulos, *Real Time Headway Predictions in Urban Rail Systems and Implications for Service Control*, 3 October 2025 | 5 | 4 | 2 | 4 | 2 | 17 | Defines the future headway module and the data structure required for 15-60 minute prediction |
| 3 | Xing et al., *Joint Optimization of Passenger Flow Control and Train Skip-Stopping for Overcrowded Metro Lines*, 28 January 2026 | 5 | 5 | 3 | 3 | 1 | 17 | Defines congestion interventions, state variables and evaluation metrics for the simulation stage |

The license score combines access to the paper with availability of reusable implementation code. The first paper states that supporting material is available on request and gives no open-source code license. The second is publicly readable on arXiv but provides no licensed repository. The third article is CC BY 4.0, while its implementation code is not published. Deployment is scored in the opposite direction to difficulty: five is easiest.

### 2.1 Passenger flow prediction with GPR

The 2025 GPR paper combines signal decomposition, dimensionality reduction, station clustering and Gaussian processes. Its main value for this project is the probabilistic regression component. CEEMDAN, PCA and GMM require a much larger station-level dataset and would be unjustified on eleven daily totals. The current prototype therefore adopts only the GPR principle and documents the omitted components.

### 2.2 Train headway prediction with ConvLSTM

The headway paper converts train-event data into a tensor whose dimensions represent time, position along the line and direction. ConvLSTM applies convolution to capture neighboring spatial patterns while the recurrent state represents how headway disturbances propagate over time. Its 30-minute input window and 15-minute direct forecast provide a practical starting point for a future MTR dataset. This model cannot be trained from the immigration dataset because that dataset has no train identity, station event time, direction-specific trajectory or headway measurement.

### 2.3 Congestion control with multi-agent reinforcement learning

The 2026 study treats stations and intervention types as cooperating agents. The simulation state includes passengers waiting to enter, transfer passengers and platform queues. Actions control entry rates, transfer flow and skip-stop decisions. The reward penalizes excess passengers and waiting time. This is relevant to the dissertation outcome, but it is not an appropriate first implementation: it requires a calibrated simulator, train capacity, station topology, passenger arrival distributions and operational constraints.

## 3 Data

The data source is the Hong Kong Immigration Department daily passenger-traffic CSV. For each date, the prototype sums all control points separately for `Arrival` and `Departure`. Total flow is the sum of those two series. Only observations from 1 to 11 November 2026 are supplied to the model.

The dataset has three important limitations. First, a clearance record is a passenger trip rather than a unique person. Second, control points are not MTR stations. Third, daily aggregation cannot represent peak-period platform congestion. These limitations restrict the current result to a forecasting-method demonstration.

## 4 Gaussian Process Regression

GPR is supervised machine learning. Instead of assuming a fixed linear equation, it places a probability distribution over possible functions. Observed passenger counts update that distribution. For a future date, the model returns both the expected count and the uncertainty around it.

For observations `y` at dates `X`, GPR assumes:

`y = f(X) + noise`, where `f` follows a Gaussian process.

The kernel determines which dates should have similar passenger counts. This prototype uses:

`k = constant × periodic_weekly + white_noise`

The periodic kernel compares dates by their position in a seven-day cycle. A date one week later is therefore strongly related to the corresponding previous weekday. The constant term learns the scale of variation, and the white-noise term prevents the model from treating every observation as exact.

The weekly period is fixed rather than optimized. Eleven observations contain only one complete week plus four repeated weekdays, which is not enough evidence for the optimizer to discover a reliable seasonal period. Fixing the period is a documented prior assumption, while amplitude, smoothness and noise remain learned parameters.

## 5 Methodology

### 5.1 Data processing

The pipeline downloads the official CSV, validates required fields, parses the date, converts passenger totals to numeric values, filters the eleven-day window, aggregates control points and verifies that all dates are present.

### 5.2 Model training

Separate GPR models are fitted for arrival and departure flows. This preserves the operational distinction between the two directions. The total forecast equals the sum of the two mean forecasts. The total uncertainty interval is approximated by treating their predictive errors as independent; this assumption must be revisited when more data permit a multivariate model.

### 5.3 Validation

Random train-test splitting is not used because it would allow future observations to influence past predictions. Expanding-window validation starts with days 1-7 and predicts day 8. The observed day 8 is then added before predicting day 9. The process continues through day 11. GPR is compared with a same-weekday seasonal baseline using MAE, RMSE and MAPE.

Four held-out forecasts cannot establish generalization across seasons, holidays or service disruptions. The metrics describe only this exploratory window.

## 6 Results

The fitted models produce the following three-day forecasts:

| Date | Arrival | Departure | Total | Approximate total 95% interval |
|---|---:|---:|---:|---:|
| 12 November 2026 | 438,031 | 450,571 | 888,601 | 819,618 to 957,585 |
| 13 November 2026 | 473,567 | 567,440 | 1,041,007 | 971,983 to 1,110,031 |
| 14 November 2026 | 562,333 | 669,100 | 1,231,433 | 1,162,449 to 1,300,416 |

In the four expanding-window validation forecasts, arrival GPR achieved MAE 22,994 and MAPE 4.52%, compared with MAE 24,660 and MAPE 4.86% for Seasonal Naive. Departure GPR achieved MAE 20,353 and MAPE 4.22%, compared with MAE 21,061 and MAPE 4.38%. For total flow, GPR achieved MAE 29,965, RMSE 36,733 and MAPE 3.16%, while Seasonal Naive achieved MAE 32,272, RMSE 38,259 and MAPE 3.41%. These small differences are exploratory because the validation set contains only four dates.

### 6.1 Post-forecast audit

The Immigration Department subsequently published the observations for 12-14 November. They were not used for training or hyperparameter fitting. Actual total flows were 948,856, 1,110,644 and 1,274,860 passenger trips. The absolute errors were 60,255, 69,637 and 43,427, giving a three-day MAE of 57,773 and MAPE of 5.34%. The first and third observations fell inside the approximate 95% intervals. The 13 November observation exceeded the upper bound by about 613 trips, showing that the independence assumption used to combine arrival and departure uncertainty is slightly too narrow for that date.

Results should be presented in this order:

1. Table of the eleven official observations.
2. Line chart of arrival, departure and total flows.
3. Model configuration and fitted kernel parameters.
4. Expanding-window comparison against Seasonal Naive.
5. Three-day forecast table with uncertainty intervals.
6. Error analysis describing the four validation dates individually.

## 7 Extension to MTR Operations

The next data schema should contain station, timestamp, entry count, exit count, transfer count, platform count, train identifier, arrival and departure time, direction, capacity, dwell time, disruption flags and relevant exogenous variables. Passenger forecasts should use 5-15 minute intervals rather than daily totals.

The headway module can then transform train events into time-position-direction tensors and compare a simple baseline with ConvLSTM. A simulation module should ingest predicted arrivals and headways, update queues and train loads, and evaluate no-control, holding, entry-control and skip-stop scenarios. Primary outcomes should include passenger-flow MAE, headway RMSE, maximum platform occupancy, excess passengers and average waiting time.

## 8 Discussion

The simplified GPR stack is technically consistent with the available evidence. It avoids an LSTM that would require more observations than the study contains, keeps deployment light by removing PyTorch, and expresses forecast uncertainty. Its main weakness is not the algorithm but the dataset: daily border-clearance data cannot validate real-time metro congestion claims.

The later ConvLSTM and reinforcement-learning stages should not be merged into the current prototype before the corresponding operational data and calibrated simulation exist. Keeping the stages separate makes every conclusion traceable to the data that support it.

## 9 Conclusion

The present system demonstrates official-data ingestion, directional passenger-flow aggregation, periodic GPR forecasting, uncertainty estimation and time-ordered validation. It is an appropriate first research prototype for the eleven-day exercise. Completion of the dissertation objectives requires new MTR data, a separate headway model and a calibrated congestion-control simulation.

## References

1. Song, Z., Feng, S. and Li, G. (2025). Urban Rail Transit Passenger Flow Prediction with Gaussian Process Regression Based on CEEMDAN. *Journal of Transportation Engineering, Part A: Systems*, 151(9). https://doi.org/10.1061/JTEPBS.TEENG-8981
2. Usama, M. and Koutsopoulos, H. (2025). Real Time Headway Predictions in Urban Rail Systems and Implications for Service Control: A Deep Learning Approach. arXiv:2510.03121. https://arxiv.org/abs/2510.03121
3. Xing, S., Guo, J., Qin, Y., Jia, L., Wang, Y. and Jiang, S. (2026). Joint Optimization of Passenger Flow Control and Train Skip-Stopping for Overcrowded Metro Lines: A Multi-Agent Reinforcement Learning Approach. *Urban Rail Transit*, 12, 38-66. https://doi.org/10.1007/s40864-025-00265-5
4. Hong Kong Immigration Department. Statistics on Daily Passenger Traffic. https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic
5. scikit-learn. Gaussian Process Regression documentation. https://scikit-learn.org/stable/modules/gaussian_process.html

