# Micro Gas Turbine Electrical Energy Prediction

- **Rows:** 71,225 total (train: 52,940 across 8 runs, test: 18,285 across 2 runs)
- **Resolution:** Variable (~10 Hz)
- **Source:** UCI ML Repository #994
- **Target column:** `el_power` (electrical output power, Watts)
- **Time column:** `time` (seconds)
- **Separator:** comma (CSV)
- **Missing values:** None
- **Noise type:** Real small gas turbine — non-constant lag between control input and output, irregular transition dynamics, mechanical + electrical measurement noise
- **Files:** `train/ex_1.csv`, `ex_9.csv`, `ex_20.csv`, `ex_21.csv`, `ex_23.csv`, `ex_24.csv` — concatenate all for full dataset
- **CLI usage (concat first):**
  ```
  python3 -c "import pandas as pd, glob; pd.concat([pd.read_csv(f) for f in glob.glob('data/micro_gas_turbine/**/*.csv', recursive=True)]).to_csv('data/micro_gas_turbine/combined.csv', index=False)"
  rgan-train --csv data/micro_gas_turbine/combined.csv --target el_power
  ```
