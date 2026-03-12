# MetroPT-3 Air Compressor Dataset

- **Rows:** 1,516,948
- **Resolution:** ~0.1 Hz (every ~10 seconds)
- **Source:** UCI ML Repository #791
- **Target column:** `TP2` (compressor pressure, bar) or `Motor_current` (Amps)
- **Time column:** `timestamp` (ISO datetime string)
- **Missing values:** None reported
- **Noise type:** Real metro train air compressor — mechanical wear, load transitions, valve state changes, electrical interference
- **Other columns:** TP3, H1, DV_pressure, Reservoirs, Oil_temperature, COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses
- **CLI usage:**
  ```
  rgan-train --csv "data/metropt3/MetroPT3(AirCompressor).csv" --target TP2
  ```
