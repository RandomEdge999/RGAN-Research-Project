# UCI Beijing PM2.5 — US Embassy Station

- **Rows:** 43,824
- **Resolution:** Hourly
- **Period:** Jan 2010 – Dec 2014 (5 years)
- **Source:** UCI ML Repository #381
- **Target column:** `pm2.5` (μg/m³)
- **Time columns:** `year`, `month`, `day`, `hour` — split, need combining
- **Missing values:** Yes (NaN), especially at start of series
- **Noise type:** Single point sensor — dropout events, haze spike bursts, calibration gaps
- **Note:** datetime column needs preprocessing (combine year/month/day/hour into single column)
