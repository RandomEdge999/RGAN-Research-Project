# USGS Napa River Streamflow

- **Rows:** 70,080
- **Resolution:** 15-minute
- **Period:** 2023-01-01 01:00 to 2025-01-01 00:45
- **Source:** USGS NWIS Instantaneous Values
- **Source URL:** https://waterdata.usgs.gov/nwis/uv
- **Target column:** `streamflow_cfs`
- **Time column:** `time`
- **File:** `usgs_napa_river_streamflow.csv`
- **Missing values:** None
- **Noise type:** River flow variability, storm pulses, hydrologic measurement noise
- **Note:** Site: 11458000 (Napa River near Napa, CA)
- **Note:** Parameter: 00060 discharge (cubic feet per second)
- **CLI usage:**
  ```
  rgan-train --csv data/usgs_napa_river_streamflow/usgs_napa_river_streamflow.csv --target streamflow_cfs --time_col time
  ```
