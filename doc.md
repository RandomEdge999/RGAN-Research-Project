# Two-Stage RGAN Code Map

This is a quick map of the exact code blocks that implement the two-stage algorithm from the draft.

This implementation follows the paper's two-stage idea closely. First it trains a normal regression model to learn the predictable part of the series, then it trains a WGAN only on the residual noise that the regression model could not explain.

At inference time, the code either uses the clean regression output `f̂(X)` for deterministic evaluation or adds a sampled residual `G(z)` to produce the hybrid forecast `f̂(X) + G(z)`.

## Core Algorithm Blocks

These are the main blocks for the actual algorithm. This is the closest translation of Algorithm 1 into Python.

- `src/rgan/rgan_torch.py:1181-1459`
  Main two-stage training function: `train_two_stage()`

- `src/rgan/rgan_torch.py:1225-1296`
  Stage 1: train regression model `f̂(X)` using MSE

- `src/rgan/rgan_torch.py:1298-1309`
  Compute residuals: `R = Y - f̂(X)`

- `src/rgan/rgan_torch.py:1311-1429`
  Stage 2: train WGAN on residuals only

- `src/rgan/rgan_torch.py:1431-1459`
  Final evaluation and return of the two-stage bundle

- `src/rgan/rgan_torch.py:1117-1147`
  Hybrid inference: deterministic `f̂(X)` or stochastic `f̂(X) + G(z)`

- `src/rgan/rgan_torch.py:1150-1178`
  Validation Wasserstein gap used for Stage 2 early stopping

## Core Model Definitions

These are the model definitions used by the two-stage pipeline. If the question is what `f̂`, `G`, and `D` are in code, this is where they live.

- `src/rgan/models_torch.py:498-542`
  `RegressionModel`

- `src/rgan/models_torch.py:545-606`
  `ResidualGenerator`

- `src/rgan/models_torch.py:609-620`
  `build_regression_model()`

- `src/rgan/models_torch.py:623-633`
  `build_residual_generator()`

- `src/rgan/models_torch.py:636-665`
  `build_residual_discriminator()`

## Data Preparation

This is where the raw CSV gets turned into the actual training data. These blocks define how `X` and `Y` are built before training starts.

- `src/rgan/data.py:26-104`
  Load CSV, detect target/time column, optional resampling

- `src/rgan/data.py:107-170`
  Interpolate missing values and standardize train/test data

- `src/rgan/data.py:184-212`
  Build univariate `(X, Y)` windows

- `src/rgan/data.py:215-245`
  Build multivariate `(X, Y)` windows with covariates

- `src/rgan/scripts/run_training.py:136-183`
  Split windowed data into train/validation sets

## Where Two-Stage Is Activated

These lines show where the code actually switches into the two-stage path. They are the quickest way to show that the paper version is the one being run.

- `src/rgan/config.py:125-129`
  Config fields for the two-stage pipeline:
  `pipeline`, `regression_epochs`, `regression_lr`, `regression_patience`

- `src/rgan/scripts/run_training.py:1898-1929`
  Build `F_hat`, `G_residual`, `D_residual`

- `src/rgan/scripts/run_training.py:2052-2072`
  Call `train_two_stage()` and save two-stage checkpoints

## Inference / Augmentation

This is the post-training path. It shows that the final synthetic forecast is built as `f̂(X) + G(z)`.

- `src/rgan/scripts/run_augmentation.py:290-347`
  Generate synthetic targets from loaded RGAN bundle

- `src/rgan/scripts/run_augmentation.py:296-316`
  Two-stage synthetic forecast path: `f̂(X) + G(z)`

## Loading Saved Two-Stage Runs

These blocks handle reloading a trained two-stage run. The loader detects `two_stage` and rebuilds the regression model plus the residual GAN bundle correctly.

- `src/rgan/scripts/run_training.py:344-363`
  Detect whether a run is `joint` or `two_stage`

- `src/rgan/scripts/run_training.py:435-540`
  Load saved two-stage models from disk

- `src/rgan/scripts/run_training.py:616-655`
  Unified prediction path after loading

## Draft Algorithm 1 to Code Mapping

This is the one-to-one map from the draft to the implementation. If a step in the draft comes up, you can jump straight to the matching code block below.

- Draft steps 1-2, construct lagged dataset and split data
  - `src/rgan/data.py:184-245`
  - `src/rgan/scripts/run_training.py:136-183`

- Draft steps 3-4, train regression model `f̂(X)`
  - `src/rgan/models_torch.py:498-542`
  - `src/rgan/rgan_torch.py:1225-1296`

- Draft step 5, generate predictions
  - `src/rgan/rgan_torch.py:1302-1306`

- Draft step 6, compute residuals
  - `src/rgan/rgan_torch.py:1298-1309`

- Draft steps 7-9, initialize and train residual WGAN
  - `src/rgan/models_torch.py:545-606`
  - `src/rgan/models_torch.py:636-665`
  - `src/rgan/scripts/run_training.py:1898-1929`

- Draft steps 10-20, critic and generator updates
  - `src/rgan/rgan_torch.py:1342-1429`

- Draft steps 21-26, hybrid forecast generation
  - `src/rgan/rgan_torch.py:1117-1147`
  - `src/rgan/scripts/run_augmentation.py:296-316`

## Minimal Set To Show In A Meeting

If time is short, open only these blocks. Together they show the data setup, the two-stage models, the training loop, and the final hybrid forecast path.

- `src/rgan/rgan_torch.py:1225-1429`
- `src/rgan/rgan_torch.py:1117-1147`
- `src/rgan/models_torch.py:498-606`
- `src/rgan/models_torch.py:636-665`
- `src/rgan/data.py:184-245`
- `src/rgan/scripts/run_training.py:1898-1929`
- `src/rgan/scripts/run_training.py:2052-2072`
- `src/rgan/scripts/run_augmentation.py:296-316`

## Note

The draft writes the method in single-step notation `x_(t+1)`. The code generalizes that to horizon `H`, so `Y`, residuals, and generated outputs are sequences shaped like `(H, 1)`.
