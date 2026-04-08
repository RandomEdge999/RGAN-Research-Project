# NOAA CO-OPS Seattle Water Level

- **Rows:** 87,840
- **Resolution:** 6-minute
- **Period:** 2024-01-01 00:00 to 2024-12-31 23:54
- **Source:** NOAA CO-OPS Data API
- **Source URL:** https://api.tidesandcurrents.noaa.gov/api/prod/
- **Target column:** `water_level`
- **Time column:** `time`
- **File:** `noaa_coops_seattle_water_level.csv`
- **Missing values:** None
- **Noise type:** Tides, estuarine forcing, storms, sensor disturbance
- **Note:** Station: 9447130 (Seattle, Washington)
- **Note:** Datum: Mean Sea Level (MSL)
- **CLI usage:**
  ```
  rgan-train --csv data/noaa_coops_seattle_water_level/noaa_coops_seattle_water_level.csv --target water_level --time_col time
  ```
