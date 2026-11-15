# Real Time Analytics of Passenger Flow and Train Headway
## Technical Report on the Eleven Day Gaussian Process Forecasting Study

**Data period:** 1–11 November 2026  
**Forecast period:** 12–14 November 2026  
**Prepared:** 15 November 2026  
**Project stage:** Public data forecasting prototype

> **Scope note:** This document is a **separate eleven-day Gaussian Process Regression (GPR) study** derived from the supplied report. The repository's existing Streamlit V2 implementation uses Seasonal Naive, Random Forest and LSTM, not the GPR architecture described here. The source PDF/DOCX contained two figures; their underlying values, titles and descriptions are retained below, but the image artwork is not embedded in this Markdown copy.

## Executive Summary

This report implements a three day passenger flow forecast using the eleven daily observations specified for the exercise. The model uses official Hong Kong Immigration Department clearance records and estimates arrival, departure and total daily flows. A Gaussian Process Regression model with a fixed seven day periodic kernel is used because it can represent the expected weekly pattern, operate with a very small dataset and return prediction uncertainty.

The model forecasts total flows of **888,601**, **1,041,007** and **1,231,433** trips for 12, 13 and 14 November. Expanding window validation over days 8 to 11 gives a total MAPE of **3.16 percent**, compared with **3.41 percent** for the same weekday baseline. The subsequently published values give a separate three day MAPE of **5.34 percent**. These results support the model as a small academic prototype, but the public data cannot measure MTR station crowding or train headway.

The full project should therefore proceed in stages. The current GPR model demonstrates official data ingestion and probabilistic passenger forecasting. A later headway module requires train event or AVL data, and congestion control requires a calibrated station and train simulation.

## 1 Project Scope and Research Question

The project aims to predict passenger flow and train headway, identify congestion and evaluate mitigation strategies. The available public dataset supports only the first part of that objective at an aggregate daily level. It records immigration clearance trips at Hong Kong control points rather than station entries, platform occupancy, train loads or train movements.

**Research question:** Can a supervised probabilistic model use the official observations from 1 to 11 November 2026 to forecast arrival, departure and total clearance flow for the following three days while reporting uncertainty?

### 1.1 Deliverables

- A verified table of eleven daily observations, separated into arrival, departure and total flow.
- Separate GPR models for arrival and departure, with their forecasts combined into total flow.
- Time ordered validation against a same weekday baseline using MAE, RMSE and MAPE.
- A three day forecast with approximate 95 percent prediction intervals.
- A clear boundary between the public data prototype and later MTR headway and congestion modules.

## 2 Literature Review and Technology Selection

Three recent studies were selected because they cover the technical stages required by the project: passenger flow prediction, train headway prediction and congestion control. Comparison uses project fit, recency, license and reproducibility, technical stack and deployment difficulty. A higher deployment score means easier implementation.

**Table 1. Selected academic references**

| Rank | Study | Technical focus | Score | Decision |
|---|---|---|---|---|
| 1 | Song et al. 2025 | GPR passenger flow | 18/25 | Use GPR now; omit CEEMDAN, PCA and GMM |
| 2 | Usama and Koutsopoulos 2025 | ConvLSTM headway | 17/25 | Use after train event data become available |
| 3 | Xing et al. 2026 | Multi agent RL control | 17/25 | Use to design the future simulation stage |

### 2.1 Passenger Flow Forecasting

Song, Feng and Li combine CEEMDAN signal decomposition, principal component analysis, Gaussian mixture clustering and GPR. Their workflow is designed for a larger urban rail dataset. Only the GPR component is suitable for the current exercise. Eleven daily observations cannot support defensible signal decomposition, station clustering or high dimensional feature reduction.

### 2.2 Train Headway Prediction

Usama and Koutsopoulos represent rail operations as a time, position and direction tensor and apply ConvLSTM. Convolution captures relationships between adjacent locations, while the recurrent state captures the propagation of delays over time. This method requires train identities, event times, locations and directions. None of those variables appears in the immigration dataset, so the method is reserved for the later headway module.

### 2.3 Congestion Control

Xing and colleagues model stations and control measures as cooperating reinforcement learning agents. The approach can coordinate entry control, transfer control and skip stopping, but it needs a calibrated simulator, passenger queues, train capacity, station topology and operational constraints. It is a useful reference for the final project stage rather than a model that can be trained from eleven aggregate observations.

## 3 Data and Preprocessing

The source is the Hong Kong Immigration Department **Statistics on Daily Passenger Traffic** dataset. For each date, all control points are summed separately for Arrival and Departure. Total flow is the sum of those directional series. The unit is a passenger clearance trip; repeated trips by the same person are counted separately.

**Table 2. Official observations used for model fitting**

| Date | Arrival | Departure | Total |
|---|---:|---:|---:|
| 2026-11-01 | 660,842 | 554,957 | 1,215,799 |
| 2026-11-02 | 477,368 | 445,864 | 923,232 |
| 2026-11-03 | 428,272 | 424,441 | 852,713 |
| 2026-11-04 | 431,514 | 432,834 | 864,348 |
| 2026-11-05 | 435,239 | 446,879 | 882,118 |
| 2026-11-06 | 473,273 | 565,189 | 1,038,462 |
| 2026-11-07 | 560,712 | 682,295 | 1,243,007 |
| 2026-11-08 | 685,204 | 513,245 | 1,198,449 |
| 2026-11-09 | 491,569 | 443,329 | 934,898 |
| 2026-11-10 | 451,557 | 436,862 | 888,419 |
| 2026-11-11 | 468,304 | 460,409 | 928,713 |

**Figure 1. Daily arrival, departure and total clearance flow.** The original report plots the three time series for 1–11 November; all underlying values are included in Table 2.

### 3.1 Data Checks

1. Parse the official day-month-year date field and sort observations chronologically.
2. Convert passenger totals to numeric values and reject missing required fields.
3. Verify that every date from 1 to 11 November is present.
4. Aggregate control points only after separating arrival and departure records.
5. Exclude all observations after 11 November from training and hyperparameter fitting.

## 4 Gaussian Process Regression

Gaussian Process Regression is supervised machine learning. It learns a probability distribution over functions that could explain observed passenger counts. For a future date, it returns a posterior mean and posterior standard deviation. The mean is the point forecast; standard deviation expresses uncertainty under the fitted model.

$$
y(t)=f(t)+\epsilon,\quad f(t)\sim GP(m(t),k(t,t'))
$$

The kernel determines how strongly two dates are related. This study uses a constant amplitude multiplied by a periodic kernel, plus a white noise term:

$$
k(t,t')=\sigma_f^2\exp\left(-\frac{2\sin^2(\pi|t-t'|/7)}{\ell^2}\right)+\sigma_n^2\delta(t,t')
$$

**Table 3. Interpretation of the GPR kernel**

| Component | Parameter | Interpretation |
|---|---|---|
| Constant amplitude | $\sigma_f^2$ | Learns scale of passenger flow variation |
| Seven day periodic term | period = 7 | Relates each weekday to the same weekday one week later |
| Length scale | $\ell$ | Controls how quickly similarity changes within the weekly cycle |
| White noise | $\sigma_n^2$ | Represents unexplained observation noise |

### 4.1 Why the Weekly Period Is Fixed

The eleven observations contain one complete week and four repeated weekdays. This is not enough to estimate a seasonal period reliably. The period is therefore fixed at seven days as a transparent modelling assumption. The model still estimates signal amplitude, within-cycle smoothness and noise level from the data.

### 4.2 Separate Directional Models

Arrival and departure have different daily profiles, so each direction is modelled separately. Their posterior means are added to obtain total flow. The total interval is approximated by adding predictive variances and taking the square root. This assumes independent directional errors, which is a limitation of the small-sample design.

$$
\mu_{total}=\mu_{arrival}+\mu_{departure},\qquad
\sigma_{total}=\sqrt{\sigma_{arrival}^2+\sigma_{departure}^2}
$$

## 5 Experimental Method

### 5.1 Expanding Window Validation

Random splitting is unsuitable for a time series because it can place future observations in the training set. Expanding window validation preserves time order. Days 1 to 7 predict day 8. The observed day 8 is then added before predicting day 9. The process continues through day 11, producing four out-of-sample predictions.

### 5.2 Baseline and Metrics

The baseline predicts each day using the value recorded exactly seven days earlier. GPR and baseline are evaluated on the same four dates. MAE measures average absolute error in passenger trips, RMSE gives more weight to large errors and MAPE expresses average absolute error as a percentage of observed flow.

$$
MAE=\operatorname{mean}(|y-\hat y|),\quad RMSE=\sqrt{\operatorname{mean}((y-\hat y)^2)},\quad MAPE=100\operatorname{mean}(|y-\hat y|/y)
$$

### 5.3 Technology Stack

**Table 4. Revised technology stack**

| Layer | Technology | Role |
|---|---|---|
| Data retrieval | Requests | Download official CSV |
| Data processing | Pandas and NumPy | Parse dates, validate records and aggregate directions |
| Machine learning | scikit-learn GPR | Fit periodic model and produce uncertainty |
| Application | Streamlit | Present fixed-study workflow |
| Charts | Plotly | Interactive history, forecast and interval figures |
| Testing | unittest | Check period, validation window and interval ordering |

The earlier Random Forest and PyTorch LSTM components were removed **from the GPR study design**. A 28-day LSTM input and a 30-day test period cannot be constructed from eleven observations. PyTorch also adds a large deployment dependency without addressing the data shortage. This statement refers to the proposed GPR study, **not to the separate existing V2 app in this repository**.

## 6 Results

**Table 5. Three day GPR forecast**

| Date | Arrival | Departure | Total | Approximate 95% interval |
|---|---:|---:|---:|---|
| 2026-11-12 | 438,031 | 450,571 | 888,601 | 819,618 to 957,585 |
| 2026-11-13 | 473,567 | 567,440 | 1,041,007 | 971,983 to 1,110,031 |
| 2026-11-14 | 562,333 | 669,100 | 1,231,433 | 1,162,449 to 1,300,416 |

**Table 6. Expanding window validation results**

| Series | Model | MAE | RMSE | MAPE (%) |
|---|---|---:|---:|---:|
| Arrival | GPR | 22,994 | 24,779 | 4.52 |
| Arrival | Seasonal Naive | 24,660 | 25,937 | 4.86 |
| Departure | GPR | 20,353 | 25,575 | 4.22 |
| Departure | Seasonal Naive | 21,061 | 25,792 | 4.38 |
| Total | GPR | 29,965 | 36,733 | 3.16 |
| Total | Seasonal Naive | 32,272 | 38,259 | 3.41 |

GPR has lower error than the same-weekday baseline for all three series. For total flow, MAPE falls from 3.41 percent to 3.16 percent. The difference is small and validation contains only four predictions, so results should be reported as exploratory evidence rather than a stable estimate of future accuracy.

**Figure 2. Forecast intervals and observations released after forecasting.** The original report illustrates the GPR total forecast, its approximate 95% prediction intervals, and actual values released subsequently. The underlying values are in Tables 5 and 7.

### 6.1 Post Forecast Audit

**Table 7. Evaluation using data published after the forecast**

| Date | Forecast | Actual | Absolute error | APE (%) | Inside interval |
|---|---:|---:|---:|---:|---|
| 2026-11-12 | 888,601 | 948,856 | 60,255 | 6.35 | Yes |
| 2026-11-13 | 1,041,007 | 1,110,644 | 69,637 | 6.27 | No |
| 2026-11-14 | 1,231,433 | 1,274,860 | 43,427 | 3.41 | Yes |

The three-day audit gives an MAE of 57,773 trips and MAPE of 5.34 percent. Observations for 12 and 14 November fall inside the approximate interval. The 13 November value is 613 trips above the upper limit, indicating that the independent-error assumption produces a slightly narrow total interval for that date. These observations were not used for model fitting.

## 7 System Architecture and Deployment

**Table 8. Processing and presentation pipeline**

| Stage | Component | Output |
|---|---|---|
| 1 | Official CSV | Daily clearance records |
| 2 | Validation and aggregation | Arrival and departure series for 1–11 November |
| 3 | Two periodic GPR models | Directional means and standard deviations |
| 4 | Aggregation | Total mean and approximate total interval |
| 5 | Streamlit interface | Tables, charts, metrics and project limitations |

The proposed GPR application has a light deployment profile: Python, scikit-learn, Pandas, NumPy, Requests, Plotly and Streamlit. It would not require a GPU or PyTorch. Because the study dataset is small, model training cost is negligible for a demonstration deployment. **This is a proposal described in the report; the current repository app remains a separate multi-model implementation.**

## 8 Extension to Train Headway and Congestion Control

### 8.1 Required Operational Data

A real-time MTR study should use 5–15 minute station observations and train event data. The minimum schema should include station, timestamp, entries, exits, transfers, platform count, train identifier, arrival time, departure time, direction, capacity, dwell time and disruption indicators.

### 8.2 Headway Module

Headway is the time between consecutive trains at the same location and in the same direction. A first operational model should compare a persistence baseline with ConvLSTM or another spatiotemporal model. Input can be organised as a time-by-line-position-by-direction tensor. Performance should be measured at several forecast horizons, such as 15, 30 and 60 minutes.

### 8.3 Congestion Simulation

The simulation should receive predicted passenger arrivals and headways, update platform queues and train loads, and test control policies. Initial policies should include no control, train holding and station entry control. Skip stopping should be added only after simulation represents transfer demand and service constraints.

**Table 9. Recommended evaluation measures for the full project**

| Area | Measures | Purpose |
|---|---|---|
| Forecasting | Passenger MAE and MAPE; headway RMSE | Prediction accuracy |
| Crowding | Maximum platform occupancy; excess passengers | Congestion severity |
| Passenger service | Average and percentile waiting time | Passenger impact |
| Operations | Headway regularity; train utilisation | Service stability |

## 9 Limitations and Risks

**Table 10. Main technical risks and controls**

| Risk | Effect | Control |
|---|---|---|
| Only eleven training days | Seasonal and holiday behaviour cannot be established | Report validation as exploratory and retain uncertainty intervals |
| Border clearance data | Results may be misinterpreted as MTR station crowding | Use clearance trip terminology and separate MTR extension |
| Fixed weekly period | Assumption may fail during holidays or disruptions | Re-estimate and compare kernels when longer data become available |
| Independent directional errors | Total interval may be too narrow | Use a multivariate model or empirical interval calibration with more data |
| No external variables | Weather, events and disruptions are omitted | Add verified exogenous features only after expanding dataset |

## 10 Conclusion

The implemented model, as described in the source report, is consistent with the eleven-day exercise. It uses supervised machine learning, respects time order, represents the expected weekly cycle and reports uncertainty. The GPR model slightly outperforms the same-weekday baseline in four expanding-window tests and produces a three-day forecast that can be checked against subsequently released official data.

The main constraint is the dataset rather than the computational method. Daily immigration-clearance totals cannot validate claims about real-time MTR congestion or train headway. Completing the full project requires authorised station and train event data, a separate headway model and a calibrated control simulation.

## References

1. Hong Kong Immigration Department. *Statistics on Daily Passenger Traffic.* https://data.gov.hk/en-data/dataset/hk-immd-set5-statistics-daily-passenger-traffic
2. Song, Z., Feng, S. and Li, G. (2025). *Urban Rail Transit Passenger Flow Prediction with Gaussian Process Regression Based on CEEMDAN.* Journal of Transportation Engineering Part A Systems, 151(9). https://doi.org/10.1061/JTEPBS.TEENG-8981
3. Usama, M. and Koutsopoulos, H. (2025). *Real Time Headway Predictions in Urban Rail Systems and Implications for Service Control A Deep Learning Approach.* arXiv 2510.03121. https://arxiv.org/abs/2510.03121
4. Xing, S., Guo, J., Qin, Y., Jia, L., Wang, Y. and Jiang, S. (2026). *Joint Optimization of Passenger Flow Control and Train Skip Stopping for Overcrowded Metro Lines A Multi Agent Reinforcement Learning Approach.* Urban Rail Transit, 12, 38–66. https://doi.org/10.1007/s40864-025-00265-5
5. scikit-learn developers. *Gaussian Process Regression documentation.* https://scikit-learn.org/stable/modules/gaussian_process.html

## Appendix A. Reproducibility Specification

**Table A1. Fixed experimental settings**

| Setting | Value |
|---|---|
| Training window | 2026-11-01 to 2026-11-11 inclusive |
| Forecast window | 2026-11-12 to 2026-11-14 inclusive |
| Targets | Arrival, Departure and Total |
| Estimator | GaussianProcessRegressor |
| Kernel | ConstantKernel × ExpSineSquared (period 7) + WhiteKernel |
| Normalisation | normalize_y = True |
| Optimizer restarts | 10 |
| Random seed | 42 |
| Validation | Four expanding-window one-step forecasts |
| Baseline | Observation exactly seven days earlier |
