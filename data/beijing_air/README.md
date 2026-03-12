# UCI Beijing Multi-Site Air Quality Data (PRSA 2017)

- **Rows:** 35,064 per station × 12 stations = 420,768 total
- **Resolution:** Hourly
- **Period:** Mar 2013 – Feb 2017 (4 years)
- **Source:** UCI ML Repository #501
- **Target column:** `PM2.5` (μg/m³)
- **Time columns:** `year`, `month`, `day`, `hour` — split, need combining
- **Missing values:** Yes, as NaN throughout
- **Noise type:** Real outdoor sensor arrays — cross-sensitivity drift, mechanical failure, weather interference
- **Stations:** Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong
- **Recommended station:** Dongsi (urban center, most complete)
- **Note:** datetime column needs preprocessing (combine year/month/day/hour into single column)
