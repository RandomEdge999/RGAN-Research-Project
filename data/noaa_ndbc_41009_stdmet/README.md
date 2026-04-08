# NOAA NDBC Buoy 41009 Standard Meteorology

- **Rows:** 52,692
- **Resolution:** 10-minute
- **Period:** 2024-01-01 00:00 to 2024-12-31 23:50
- **Source:** NOAA National Data Buoy Center
- **Source URL:** https://www.ndbc.noaa.gov/historical_data.shtml
- **Target column:** `wvht`
- **Time column:** `time`
- **File:** `noaa_ndbc_41009_stdmet.csv`
- **Missing values:** None
- **Noise type:** Open-ocean wave and meteorological variability, tropical systems, sensor dropout
- **Note:** Station: 41009 (Canaveral East buoy)
- **Note:** Recommended target: WVHT (significant wave height)
- **CLI usage:**
  ```
  rgan-train --csv data/noaa_ndbc_41009_stdmet/noaa_ndbc_41009_stdmet.csv --target wvht --time_col time
  ```
