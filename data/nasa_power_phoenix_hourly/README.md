# NASA POWER Phoenix Hourly Meteorology

- **Rows:** 70,128
- **Resolution:** Hourly
- **Period:** 2017-01-01 00:00 to 2024-12-31 23:00
- **Source:** NASA POWER API
- **Source URL:** https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- **Target column:** `T2M`
- **Time column:** `time`
- **File:** `nasa_power_phoenix_hourly.csv`
- **Missing values:** None
- **Noise type:** Dry-climate temperature swings, convective events, reanalysis uncertainty
- **Note:** Location: Phoenix, Arizona, USA
- **Note:** Recommended target: T2M (2m air temperature)
- **CLI usage:**
  ```
  rgan-train --csv data/nasa_power_phoenix_hourly/nasa_power_phoenix_hourly.csv --target T2M --time_col time
  ```
