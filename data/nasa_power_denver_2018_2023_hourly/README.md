# NASA POWER Denver Hourly Meteorology (2018-2023)

- **Rows:** 52,584
- **Resolution:** Hourly
- **Period:** 2018-01-01 00:00 to 2023-12-31 23:00
- **Source:** NASA POWER API
- **Source URL:** https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- **Target column:** `T2M`
- **Time column:** `time`
- **File:** `nasa_power_denver_2018_2023_hourly.csv`
- **Missing values:** None
- **Noise type:** Natural weather variability, fronts, seasonal cycles, atmospheric uncertainty
- **Note:** Location: Denver, Colorado, USA
- **Note:** Recommended target: T2M (2m air temperature)
- **CLI usage:**
  ```
  rgan-train --csv data/nasa_power_denver_2018_2023_hourly/nasa_power_denver_2018_2023_hourly.csv --target T2M --time_col time
  ```
