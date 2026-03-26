# Wind Turbine SCADA Dataset

- **Rows:** 50,530
- **Resolution:** 10-minute
- **Period:** 2018-01-01 00:00 to 2018-12-31 23:50
- **Source:** Kaggle (Berk Erisen) / GitHub mirror
- **Source URL:** https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset
- **Target column:** `active_power_kw`
- **Time column:** `time`
- **File:** `wind_turbine_scada.csv`
- **Missing values:** None
- **Noise type:** Wind turbulence, stochastic power output, cut-in/cut-out transitions
- **Note:** Single turbine, columns renamed from original (spaces/parens removed)
- **CLI usage:**
  ```
  rgan-train --csv data/wind_turbine_scada/wind_turbine_scada.csv --target active_power_kw --time_col time
  ```
