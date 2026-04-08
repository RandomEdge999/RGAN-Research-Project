# NOAA NDBC Buoy 46026 Standard Meteorology

- **Rows:** 49,309
- **Resolution:** 10-minute
- **Period:** 2024-01-01 00:00 to 2024-12-31 23:50
- **Source:** NOAA National Data Buoy Center
- **Source URL:** https://www.ndbc.noaa.gov/historical_data.shtml
- **Target column:** `wvht`
- **Time column:** `time`
- **File:** `noaa_ndbc_46026_stdmet.csv`
- **Missing values:** None
- **Noise type:** Open-ocean wave and meteorological variability, storms, sensor dropout
- **Note:** Station: 46026 (San Francisco buoy)
- **Note:** Recommended target: WVHT (significant wave height)
- **CLI usage:**
  ```
  rgan-train --csv data/noaa_ndbc_46026_stdmet/noaa_ndbc_46026_stdmet.csv --target wvht --time_col time
  ```
