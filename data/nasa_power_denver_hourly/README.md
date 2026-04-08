# NASA POWER Denver Hourly Meteorology

- **Rows:** 70,128
- **Resolution:** Hourly
- **Period:** 2017-01-01 00:00 to 2024-12-31 23:00
- **Source:** NASA POWER API
- **Source URL:** https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- **Target column:** `T2M`
- **Time column:** `time`
- **File:** `nasa_power_denver_hourly.csv`
- **Missing values:** None
- **Noise type:** Mountain-plain weather shifts, snow/cold fronts, reanalysis uncertainty
- **Note:** Location: Denver, Colorado, USA
- **Note:** Recommended target: T2M (2m air temperature)
- **CLI usage:**
  ```
  rgan-train --csv data/nasa_power_denver_hourly/nasa_power_denver_hourly.csv --target T2M --time_col time
  ```
