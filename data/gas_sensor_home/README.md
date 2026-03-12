# Gas Sensors for Home Activity Monitoring

- **Rows:** 928,991
- **Resolution:** Variable (~1 Hz)
- **Source:** UCI ML Repository #362
- **Target column:** `R1`–`R8` (MOX sensor resistance, kΩ) or `Temp.`
- **Time column:** `time` (seconds elapsed)
- **Separator:** whitespace-separated `.dat` file
- **Missing values:** None
- **Noise type:** 8 metal-oxide sensors in real home — background odours, airflow variation, temperature/humidity fluctuations, long-term baseline drift
- **Activities:** wine presence, banana presence, background
- **Note:** No absolute datetime — `time` is seconds since experiment start. No resampling needed.
