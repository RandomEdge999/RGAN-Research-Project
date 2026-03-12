# UCI Household Electric Power Consumption

- **Rows:** 2,075,259
- **Resolution:** 1-minute
- **Period:** Dec 2006 – Nov 2010 (4 years)
- **Source:** UCI ML Repository #235
- **Target column:** `Global_active_power` (kW)
- **Separator:** semicolon (`;`)
- **Time columns:** `Date` (DD/MM/YYYY) + `Time` (HH:MM:SS) — separate columns, combine for datetime
- **Missing values:** ~26K rows (1.25%) marked as `?`
- **Noise type:** Real residential meter — appliance load spikes, voltage fluctuations, missing readings
- **CLI usage:**
  ```
  rgan-train --csv data/household_power/household_power_consumption.txt --target Global_active_power
  ```
