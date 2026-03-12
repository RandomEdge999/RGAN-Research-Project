# Things to Fix — Research Gaps & Next Steps

Last updated: 2026-03-12

---

## Current Models

| Model | Type | Architecture | Status |
|-------|------|-------------|--------|
| **RGAN** | GAN (ours) | LSTM Generator + LSTM Discriminator, noise input | Done |
| **RGAN (WGAN)** | GAN variant | Same arch, Wasserstein loss + weight clipping | Done |
| **RGAN (WGAN-GP)** | GAN variant | Same arch, Wasserstein loss + gradient penalty | Done |
| **LSTM Supervised** | Neural baseline | Single-layer LSTM, MSE loss, no GAN | Done |
| **Naive** | Statistical baseline | Repeat last observed value | Done |
| **ARIMA** | Statistical baseline | statsmodels ARIMA(1,1,1) | Done |
| **ARMA** | Statistical baseline | ARIMA with d=0 | Done |
| **Tree Ensemble** | ML baseline | GradientBoostingRegressor (sklearn) | Done |
| **DLinear** | Linear baseline | Decomposition-Linear (trend + remainder) | Done |
| **NLinear** | Linear baseline | Normalized-Linear (last-value subtraction) | Done |
| **FITS** | Frequency baseline | rFFT → low-pass → complex linear interp → iFFT (~10k params) | Done |
| **PatchTST** | Transformer baseline | Patched input + channel-independent Transformer encoder | Done |
| **iTransformer** | Transformer baseline | Inverted Transformer (attention across variates/segments) | Done |
| **TimeGAN** | GAN competitor | 4-component GAN (embedder/recovery/generator/discriminator) | Done |

---

## Priority Fixes

### ~~1. Fix Data Augmentation~~ — DONE (code fixed, needs full run)

Pipeline rewritten. Fixes applied:
- ✅ Architecture params now read correctly from metrics.json (`units_g`, `g_layers`)
- ✅ Stochastic generation via dropout noise + input perturbation (5 runs, round-robin)
- ✅ Scale mismatch auto-detection and re-normalization
- ✅ Neural models tested (LSTM, FITS, PatchTST, iTransformer) instead of statistical ones
- ✅ TimeGAN comparison integrated
- ✅ Print formatting fixed

**Status:** Code is ready. Needs a full run on Binance data (on AWS) to verify augmentation actually helps.

### ~~2. Noise Robustness Experiments~~ — DONE (code complete, needs full run)

- ✅ Noise injection loop tests all 11 models at each σ level
- ✅ `noise_robustness_table.csv` — RMSE per model per σ, with Degradation % column
- ✅ `noise_robustness.png` / `.html` — line plot of RMSE vs σ (the key paper figure)
- ✅ Smoke-tested with `--noise_levels 0,0.05,0.1` — works end-to-end

**Status:** Code is ready. Needs full run with `--noise_levels 0,0.01,0.05,0.1,0.2` and 80+ epochs on AWS.

### 3. Fix Split Date Columns in Beijing Air Quality & Beijing PM2.5

Both datasets store datetime as separate `year`, `month`, `day`, `hour` columns instead of a single datetime column. `data.py` doesn't handle this — needs a preprocessing step to combine them into a single `datetime` column before the pipeline can use them.

**Affected files:**
- `data/beijing_air/PRSA_Data_20130301-20170228/*.csv` — target: `PM2.5`
- `data/beijing_pm25/PRSA_data.csv` — target: `pm2.5`

**Fix needed in:** `src/rgan/data.py` → `load_csv_series()` — detect and combine split date columns automatically.

---

### 4. Complete TSLib Benchmarks

Only ETTh1 has been run. Need ETTh2, Weather, and Exchange for all 4 horizons.

**Action:**
```bash
rgan-benchmark --datasets ETTh2,Weather,Exchange --pred_lens 96,192,336,720
```

### 4. Multiple Seeds

All results use seed=42 only. Reviewers will reject single-seed results.

**Action:** Run key experiments with seeds {42, 123, 456, 789, 1024} and report mean ± std.

### 5. M5 Dataset — Complete Results

M5 results exist but appear partial. Need full pipeline (RGAN + all baselines + augmentation) on this second real-world dataset.

### 6. Ablation Study

Show what each component contributes:
- GAN loss vs supervised-only (RGAN vs LSTM)
- WGAN-GP vs WGAN vs standard BCE
- EMA vs no-EMA
- Different lambda_reg values (0.1, 1.0, 5.0, 10.0)
- Supervised warmup vs no warmup

---

## Models to Consider Adding

### High Priority — Modern Transformer Baselines

These are the models reviewers will compare your numbers against. You don't necessarily need to **implement** them all, but you should **cite their published numbers** from the TSLib benchmarks and compare in a table.

| Model | Year | Why it matters | Implement or cite? |
|-------|------|---------------|-------------------|
| **DLinear** | 2023 | Dead-simple linear model that embarrassed Transformers on forecasting. If you can't beat DLinear, the GAN adds no value. | **DONE** — `linear_baselines.py` |
| **PatchTST** | 2023 | Transformer with patched input, SOTA on many benchmarks | **DONE** — `patchtst.py` |
| **iTransformer** | 2024 | Inverted Transformer, current SOTA on multivariate | **DONE** — `itransformer.py` |
| **FITS** | 2024 | Frequency-domain linear, ~10k params, ICLR Spotlight | **DONE** — `fits.py` |
| **TimesNet** | 2023 | CNN-based, strong across tasks | Cite published numbers |
| **N-BEATS** | 2020 | Pure MLP, very strong univariate baseline | Cite published numbers |

### Medium Priority — Would Strengthen the Paper

| Model | Why |
|-------|-----|
| **TimeGAN** | Direct GAN competitor for time-series generation — **DONE** in `timegan.py` |
| **N-HiTS** | Improved N-BEATS, strong univariate forecaster |
| **DeepAR** | Probabilistic forecasting baseline (Amazon), good comparison point |

### Low Priority — Nice to Have

| Model | Why |
|-------|-----|
| **Informer** | Early efficient Transformer for long-horizon, widely cited |
| **FEDformer** | Frequency-domain Transformer |
| **Autoformer** | Auto-correlation based Transformer |

### Recommendation

The real differentiator for the paper isn't beating Transformers on clean data (you probably won't), it's showing that:

1. RGAN is more robust under noise than all baselines
2. RGAN-generated synthetic data improves downstream forecasting
3. The GAN component adds value over supervised LSTM alone

For models not implemented, **cite published TSLib numbers** in the comparison table — this is standard practice.

---

## Experiment Priority Order

1. ~~Fix augmentation pipeline~~ — DONE (code ready, needs full AWS run)
2. ~~Noise robustness experiments~~ — DONE (code ready, needs full AWS run)
3. ~~Implement DLinear baseline~~ — DONE (DLinear + NLinear)
4. ~~Implement modern baselines~~ — DONE (FITS, PatchTST, iTransformer, TimeGAN)
5. **Run full Binance experiment on AWS** (80 epochs, all models, noise levels, augmentation)
6. Complete TSLib benchmarks (ETTh2, Weather, Exchange)
7. Multiple seeds for all key experiments
8. Ablation study
9. M5 full results

---

## What's Left Before Paper Submission

### Must-Have (Reviewer will reject without these)
- [ ] Full Binance results with all 14 models (80+ epochs, GPU)
- [ ] Noise robustness table + figure with 5+ noise levels
- [ ] Augmentation results showing synthetic data helps neural models
- [ ] Multi-seed results (5 seeds, mean ± std) for at least Binance + ETTh1
- [ ] TSLib benchmarks on at least 2 more datasets (ETTh2 + one of Weather/Exchange)

### Should-Have (Strengthens the paper significantly)
- [ ] Ablation study (GAN variant, lambda, EMA, warmup)
- [ ] M5 dataset results (shows generalization to retail domain)
- [ ] TimeGAN vs RGAN head-to-head comparison table

### Nice-to-Have (Polish)
- [ ] Published numbers for TimesNet, N-BEATS cited in comparison table
- [ ] Confidence intervals on all key results
- [ ] Paper-quality LaTeX tables auto-generated from CSV outputs
