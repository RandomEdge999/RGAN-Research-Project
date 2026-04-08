import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from cloud.launch import _build_hyperparameters
from rgan.config import TrainConfig
from rgan.data import (
    interpolate_and_standardize,
    load_csv_series,
    make_windows_univariate,
    make_windows_with_covariates,
)
from rgan.models_torch import (
    build_discriminator,
    build_generator,
    build_regression_model,
    build_residual_discriminator,
    build_residual_generator,
)
from rgan.plots import plot_training_curves_overlay
from rgan.rgan_torch import (
    _DEFAULT_PRELOAD_LIMIT_BYTES,
    _estimate_split_bytes,
    _gradient_penalty,
    _should_apply_critic_regularizer,
    _should_preload_to_device,
    compute_hybrid_metrics,
    train_rgan_torch,
)
from rgan.scripts.run_training import (
    _build_resume_signature,
    _detect_rgan_pipeline,
    _extract_allowed_resume_flag_changes,
    _invalidate_stages_for_resume_flag_changes,
    _load_rgan_bundle_from_run_dir,
    _predict_rgan_bundle,
    _signature_differences,
    _strip_allowed_resume_flag_changes,
    _stage_entry_complete,
)
from rgan.scripts.run_augmentation import (
    _build_parser,
    _infer_generator_replay_config,
    _infer_two_stage_replay_config,
    _generate_rgan_synthetic_targets,
    _resolve_augmentation_data_config,
    _resolve_augmentation_seed,
    _resolve_rgan_source,
    _sample_timegan_indices,
    _shuffle_mixed_indices,
    train_evaluate_classifiers,
)
from rgan.scripts.run_benchmark import _device_str
from rgan.synthetic_analysis import evaluate_discriminators
from rgan.timegan import train_timegan
from rgan.tslib_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResidualFixRegressionTests(unittest.TestCase):
    def _make_resume_args(self, csv_path: Path, **overrides):
        values = dict(
            csv=str(csv_path),
            target="value",
            time_col="timestamp",
            resample="",
            agg="last",
            train_ratio=0.8,
            val_frac=0.1,
            max_train_windows=0,
            only_models="",
            prior_results="",
            skip_classical=False,
            skip_noise_robustness=False,
            bootstrap_samples=0,
            noise_levels="0",
            L=12,
            H=3,
            batch_size=8,
            eval_every=1,
            eval_batch_size=16,
            seed=42,
            deterministic=False,
            lambda_reg=0.5,
            gan_variant="wgan-gp",
            units_g=8,
            units_d=8,
            g_layers=1,
            d_layers=1,
            lr_g=5e-4,
            lr_d=5e-4,
            label_smooth=0.0,
            grad_clip=1.0,
            dropout=0.1,
            patience=2,
            g_dense_activation="",
            d_activation="sigmoid",
            amp=False,
            ema_decay=0.0,
            wgan_gp_lambda=10.0,
            wgan_clip_value=0.01,
            use_logits=False,
            d_steps=1,
            g_steps=1,
            supervised_warmup_epochs=0,
            lambda_reg_start=0.5,
            lambda_reg_end=0.5,
            lambda_reg_warmup_epochs=1,
            adv_weight=1.0,
            instance_noise_std=0.0,
            instance_noise_decay=1.0,
            weight_decay=0.0,
            noise_dim=8,
            pipeline="joint",
            regression_epochs=10,
            regression_lr=5e-4,
            regression_patience=3,
            lambda_diversity=0.0,
            critic_reg_interval=1,
            critic_arch="tcn",
            epochs=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_wgan_gp_honors_explicit_single_critic_step(self):
        config = TrainConfig(
            L=12,
            H=3,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            units_g=8,
            units_d=8,
            g_layers=1,
            d_layers=1,
            gan_variant="wgan-gp",
            d_steps=1,
            g_steps=1,
            supervised_warmup_epochs=0,
            patience=2,
            amp=False,
            device="cpu",
            critic_arch="tcn",
            preload_to_device="never",
            compile_mode="off",
            eval_every=1,
            ema_decay=0.0,
        )
        generator = build_generator(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
        discriminator = build_discriminator(
            L=12,
            H=3,
            units=8,
            dropout=0.0,
            num_layers=1,
            activation="linear",
            critic_arch="tcn",
        )
        splits = {
            "Xtr": np.random.randn(8, 12, 1).astype(np.float32),
            "Ytr": np.random.randn(8, 3, 1).astype(np.float32),
            "Xval": np.random.randn(4, 12, 1).astype(np.float32),
            "Yval": np.random.randn(4, 3, 1).astype(np.float32),
            "Xte": np.random.randn(4, 12, 1).astype(np.float32),
            "Yte": np.random.randn(4, 3, 1).astype(np.float32),
        }

        result = train_rgan_torch(config, (generator, discriminator), splits, results_dir=".")

        self.assertEqual(config.d_steps, 1)
        self.assertEqual(len(result["history"]["D_loss"]), 1)

    def test_resume_signature_ignores_epochs_but_catches_batch_size_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy.csv"
            csv_path.write_text("timestamp,value\n1,1.0\n2,2.0\n", encoding="utf-8")

            base_args = self._make_resume_args(csv_path, epochs=100)
            same_run_new_target = self._make_resume_args(csv_path, epochs=150)
            drifted_args = self._make_resume_args(csv_path, batch_size=32)

            base_signature = _build_resume_signature(base_args)

            self.assertEqual(_signature_differences(base_signature, _build_resume_signature(same_run_new_target)), [])
            diffs = _signature_differences(base_signature, _build_resume_signature(drifted_args))
            self.assertIn("training.batch_size: 8 -> 32", diffs)

    def test_resume_signature_catches_pipeline_and_regression_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy.csv"
            csv_path.write_text("timestamp,value\n1,1.0\n2,2.0\n", encoding="utf-8")

            base_signature = _build_resume_signature(self._make_resume_args(csv_path))
            changed_signature = _build_resume_signature(
                self._make_resume_args(
                    csv_path,
                    pipeline="two_stage",
                    regression_epochs=25,
                    regression_lr=1e-3,
                    regression_patience=9,
                )
            )

            diffs = _signature_differences(base_signature, changed_signature)
            self.assertIn("training.pipeline: 'joint' -> 'two_stage'", diffs)
            self.assertIn("training.regression_epochs: 10 -> 25", diffs)
            self.assertIn("training.regression_lr: 0.0005 -> 0.001", diffs)
            self.assertIn("training.regression_patience: 3 -> 9", diffs)

    def test_resume_signature_allows_eval_flag_changes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy.csv"
            csv_path.write_text("timestamp,value\n1,1.0\n2,2.0\n", encoding="utf-8")

            previous_args = self._make_resume_args(
                csv_path,
                skip_classical=True,
                skip_noise_robustness=True,
            )
            current_args = self._make_resume_args(
                csv_path,
                skip_classical=False,
                skip_noise_robustness=False,
            )

            previous_signature = _build_resume_signature(previous_args)
            current_signature = _build_resume_signature(current_args)

            changes = _extract_allowed_resume_flag_changes(previous_signature, current_signature)
            self.assertEqual(
                changes,
                {
                    "pipeline.skip_classical": {"previous": True, "current": False},
                    "pipeline.skip_noise_robustness": {"previous": True, "current": False},
                },
            )
            self.assertEqual(
                _signature_differences(
                    _strip_allowed_resume_flag_changes(previous_signature),
                    _strip_allowed_resume_flag_changes(current_signature),
                ),
                [],
            )

    def test_eval_flag_changes_invalidate_only_dependent_stages(self):
        manifest = {
            "stages": {
                "rgan": {"status": "completed"},
                "classical_baselines": {"status": "completed"},
                "bootstrap": {"status": "completed"},
                "noise_robustness": {"status": "completed"},
                "reporting": {"status": "completed"},
            }
        }

        invalidated = _invalidate_stages_for_resume_flag_changes(
            manifest,
            {
                "pipeline.skip_classical": {"previous": True, "current": False},
                "pipeline.skip_noise_robustness": {"previous": True, "current": False},
            },
        )

        self.assertEqual(invalidated, ["classical_baselines", "noise_robustness", "reporting"])
        self.assertEqual(manifest["stages"]["rgan"]["status"], "completed")
        self.assertEqual(manifest["stages"]["bootstrap"]["status"], "completed")
        self.assertEqual(manifest["stages"]["classical_baselines"]["status"], "pending")
        self.assertEqual(manifest["stages"]["noise_robustness"]["status"], "pending")
        self.assertEqual(manifest["stages"]["reporting"]["status"], "pending")

    def test_stage_entry_complete_requires_cache_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            resume_store = Path(tmp)
            cache_dir = resume_store / "stage_cache"
            cache_dir.mkdir()
            torch.save({"ok": True}, cache_dir / "demo.pt")
            artifact = resume_store / "metrics.json"
            artifact.write_text("{}", encoding="utf-8")

            entry = {
                "status": "completed",
                "cache": "stage_cache/demo.pt",
                "artifacts": ["metrics.json"],
            }

            self.assertTrue(_stage_entry_complete(entry, resume_store))
            artifact.unlink()
            self.assertFalse(_stage_entry_complete(entry, resume_store))

    def test_resume_epoch_targets_support_same_target_and_reject_lower_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            checkpoint_dir = tmpdir / "checkpoints"
            checkpoint_dir.mkdir()
            splits = {
                "Xtr": np.random.randn(8, 12, 1).astype(np.float32),
                "Ytr": np.random.randn(8, 3, 1).astype(np.float32),
                "Xval": np.random.randn(4, 12, 1).astype(np.float32),
                "Yval": np.random.randn(4, 3, 1).astype(np.float32),
                "Xte": np.random.randn(4, 12, 1).astype(np.float32),
                "Yte": np.random.randn(4, 3, 1).astype(np.float32),
            }

            first_config = TrainConfig(
                L=12,
                H=3,
                epochs=1,
                batch_size=4,
                eval_batch_size=4,
                units_g=8,
                units_d=8,
                g_layers=1,
                d_layers=1,
                gan_variant="wgan-gp",
                d_steps=1,
                g_steps=1,
                supervised_warmup_epochs=0,
                patience=2,
                amp=False,
                device="cpu",
                critic_arch="tcn",
                preload_to_device="never",
                compile_mode="off",
                eval_every=1,
                ema_decay=0.0,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_every=1,
            )
            first_generator = build_generator(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
            first_discriminator = build_discriminator(
                L=12,
                H=3,
                units=8,
                dropout=0.0,
                num_layers=1,
                activation="linear",
                critic_arch="tcn",
            )
            first_result = train_rgan_torch(first_config, (first_generator, first_discriminator), splits, results_dir=str(tmpdir))
            self.assertEqual(first_result["resumed_from_epoch"], 0)

            checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
            self.assertTrue(checkpoint_path.exists())

            resume_generator = build_generator(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
            resume_discriminator = build_discriminator(
                L=12,
                H=3,
                units=8,
                dropout=0.0,
                num_layers=1,
                activation="linear",
                critic_arch="tcn",
            )
            same_target_config = dataclasses.replace(
                first_config,
                resume_from=str(checkpoint_path),
                checkpoint_dir=str(checkpoint_dir),
                epochs=1,
            )
            same_target_result = train_rgan_torch(
                same_target_config,
                (resume_generator, resume_discriminator),
                splits,
                results_dir=str(tmpdir),
            )
            self.assertEqual(same_target_result["resumed_from_epoch"], 1)
            self.assertEqual(same_target_result["history"]["epoch"], [1])

            lower_target_generator = build_generator(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
            lower_target_discriminator = build_discriminator(
                L=12,
                H=3,
                units=8,
                dropout=0.0,
                num_layers=1,
                activation="linear",
                critic_arch="tcn",
            )
            lower_target_config = dataclasses.replace(
                first_config,
                resume_from=str(checkpoint_path),
                checkpoint_dir=str(checkpoint_dir),
                epochs=0,
            )
            with self.assertRaises(ValueError):
                train_rgan_torch(
                    lower_target_config,
                    (lower_target_generator, lower_target_discriminator),
                    splits,
                    results_dir=str(tmpdir),
                )

    def test_cloud_launch_builds_new_performance_hyperparameters(self):
        defaults = {
            "epochs": 200,
            "batch_size": 512,
            "L": 60,
            "H": 12,
            "noise_levels": "0,0.01",
            "checkpoint_every": 10,
            "resample": "1min",
            "agg": "last",
            "units_g": 128,
            "units_d": 128,
            "g_layers": 2,
            "d_layers": 2,
            "lrG": 5e-4,
            "lrD": 5e-4,
            "lambda_reg": 0.5,
            "gan_variant": "wgan-gp",
            "wgan_gp_lambda": 10.0,
            "d_steps": 3,
            "g_steps": 1,
            "grad_clip": 1.0,
            "dropout": 0.1,
            "ema_decay": 0.999,
            "supervised_warmup_epochs": 10,
            "patience": 25,
            "bootstrap_samples": 300,
            "num_workers": 2,
            "seed": 42,
            "train_ratio": 0.8,
            "weight_decay": 0.0,
            "adv_weight": 1.0,
            "eval_every": 5,
            "eval_batch_size": 2048,
            "compile_mode": "off",
            "preload_to_device": "never",
            "critic_reg_interval": 1,
            "critic_arch": "tcn",
            "skip_classical": True,
            "skip_noise_robustness": True,
        }
        args = SimpleNamespace(
            epochs=None,
            batch_size=None,
            L=None,
            H=None,
            noise_levels=None,
            checkpoint_every=None,
            seed=None,
            gan_variant=None,
            target="index_value",
            time_col="calc_time",
            resample="5min",
            agg="mean",
            skip_classical=None,
            skip_noise_robustness=None,
            deterministic=False,
            max_train_windows=None,
            only_models="",
            prior_results="",
            compile_mode="reduce-overhead",
            preload_to_device="auto",
            critic_reg_interval=8,
            critic_arch="lstm",
        )

        hyperparameters = _build_hyperparameters(args, defaults)

        self.assertEqual(hyperparameters["batch_size"], "512")
        self.assertEqual(hyperparameters["resample"], "5min")
        self.assertEqual(hyperparameters["agg"], "mean")
        self.assertEqual(hyperparameters["compile_mode"], "reduce-overhead")
        self.assertEqual(hyperparameters["preload_to_device"], "auto")
        self.assertEqual(hyperparameters["critic_reg_interval"], "8")
        self.assertEqual(hyperparameters["critic_arch"], "lstm")
        self.assertEqual(hyperparameters["skip_classical"], "true")
        self.assertEqual(hyperparameters["skip_noise_robustness"], "true")

        args.skip_classical = False
        args.skip_noise_robustness = False
        hyperparameters = _build_hyperparameters(args, defaults)
        self.assertNotIn("skip_classical", hyperparameters)
        self.assertNotIn("skip_noise_robustness", hyperparameters)

    def test_cloud_config_uses_binance_throughput_defaults(self):
        cfg = yaml.safe_load((PROJECT_ROOT / "cloud" / "config.yaml").read_text())
        defaults = cfg["defaults"]

        self.assertEqual(defaults["batch_size"], 1024)
        self.assertEqual(defaults["resample"], "")
        self.assertEqual(defaults["preload_to_device"], "auto")
        self.assertEqual(defaults["compile_mode"], "reduce-overhead")
        self.assertEqual(defaults["critic_arch"], "tcn")
        self.assertTrue(defaults["skip_classical"])
        self.assertTrue(defaults["skip_noise_robustness"])

    def test_preload_decision_uses_device_and_size_threshold(self):
        tiny_splits = {
            "Xtr": np.zeros((32, 60, 1), dtype=np.float32),
            "Ytr": np.zeros((32, 12, 1), dtype=np.float32),
            "Xval": np.zeros((8, 60, 1), dtype=np.float32),
            "Yval": np.zeros((8, 12, 1), dtype=np.float32),
            "Xte": np.zeros((8, 60, 1), dtype=np.float32),
            "Yte": np.zeros((8, 12, 1), dtype=np.float32),
        }

        total_bytes = _estimate_split_bytes(tiny_splits)

        self.assertLess(total_bytes, _DEFAULT_PRELOAD_LIMIT_BYTES)
        self.assertTrue(_should_preload_to_device("always", torch.device("cuda:0"), total_bytes))
        self.assertTrue(_should_preload_to_device("auto", torch.device("cuda:0"), total_bytes))
        self.assertFalse(_should_preload_to_device("auto", torch.device("cpu"), total_bytes))
        self.assertFalse(_should_preload_to_device("never", torch.device("cuda:0"), total_bytes))
        self.assertFalse(
            _should_preload_to_device(
                "auto",
                torch.device("cuda:0"),
                _DEFAULT_PRELOAD_LIMIT_BYTES + 1,
            )
        )

    def test_lazy_gp_interval_fires_on_expected_critic_steps(self):
        fired_steps = [
            step for step in range(1, 17)
            if _should_apply_critic_regularizer(step, 8)
        ]

        self.assertEqual(fired_steps, [8, 16])
        self.assertTrue(_should_apply_critic_regularizer(3, 1))

    def test_build_discriminator_supports_lstm_and_tcn_critics(self):
        lstm_disc = build_discriminator(
            L=12,
            H=3,
            units=16,
            dropout=0.1,
            num_layers=2,
            activation="linear",
            layer_norm=True,
            use_spectral_norm=True,
            critic_arch="lstm",
        )
        tcn_disc = build_discriminator(
            L=12,
            H=3,
            units=16,
            dropout=0.1,
            num_layers=2,
            activation="linear",
            layer_norm=True,
            use_spectral_norm=True,
            critic_arch="tcn",
        )

        self.assertEqual(type(lstm_disc).__name__, "Discriminator")
        self.assertEqual(type(tcn_disc).__name__, "TCNDiscriminator")

        real = torch.randn(4, 15, 1)
        fake = torch.randn(4, 15, 1)
        scores = tcn_disc(real)
        self.assertEqual(tuple(scores.shape), (4, 1))

        gp = _gradient_penalty(tcn_disc, real, fake, torch.device("cpu"), 10.0)
        self.assertGreaterEqual(float(gp.detach().item()), 0.0)
        gp.backward()

        other_shape_disc = build_discriminator(
            L=7,
            H=2,
            units=8,
            dropout=0.0,
            num_layers=3,
            activation="linear",
            critic_arch="tcn",
        )
        other_scores = other_shape_disc(torch.randn(3, 9, 1))
        self.assertEqual(tuple(other_scores.shape), (3, 1))

    def test_binance_window_counts_match_raw_and_resampled_expectations(self):
        csv_path = PROJECT_ROOT / "data" / "binance" / "Binance_Data.csv"
        self.assertTrue(csv_path.exists(), msg=f"Missing Binance dataset: {csv_path}")

        df_raw, target_raw, _ = load_csv_series(
            str(csv_path),
            target="index_value",
            time_col="calc_time",
            resample="",
            agg="last",
        )
        prep_raw = interpolate_and_standardize(df_raw, target_raw, train_ratio=0.8)
        x_raw, _ = make_windows_univariate(prep_raw["scaled_train"], target_raw, 60, 12)

        df_resampled, target_resampled, _ = load_csv_series(
            str(csv_path),
            target="index_value",
            time_col="calc_time",
            resample="1min",
            agg="last",
        )
        prep_resampled = interpolate_and_standardize(df_resampled, target_resampled, train_ratio=0.8)
        x_resampled, _ = make_windows_univariate(prep_resampled["scaled_train"], target_resampled, 60, 12)

        self.assertEqual(len(df_raw), 86398)
        self.assertEqual(len(x_raw), 69047)
        self.assertEqual(len(df_resampled), 1440)
        self.assertEqual(len(x_resampled), 1081)

    def test_cached_tslib_dataset_loads_without_datasets_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            csv_path = data_dir / "ETTh1.csv"
            csv_path.write_text("date,OT\n2024-01-01,1.0\n2024-01-02,2.0\n", encoding="utf-8")

            df, target = load_dataset("ETTh1", data_dir=str(data_dir))

            self.assertEqual(target, "OT")
            self.assertEqual(list(df.columns), ["date", "OT"])
            self.assertEqual(len(df), 2)

    def test_augmentation_requires_explicit_model_or_baseline_flag(self):
        args = SimpleNamespace(
            rgan_model=None,
            results_from=None,
            auto_discover_results=False,
            allow_gaussian_baseline=False,
        )
        with self.assertRaises(FileNotFoundError):
            _resolve_rgan_source(args)

    def test_augmentation_replays_run_config_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "args": {
                            "target": "value",
                            "time_col": "timestamp",
                            "resample": "5min",
                            "agg": "mean",
                            "train_ratio": 0.7,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                target_col="auto",
                time_col="auto",
                resample="",
                agg="last",
                train_split=0.8,
            )

            cfg = _resolve_augmentation_data_config(args, run_dir)

            self.assertEqual(cfg["target_col"], "value")
            self.assertEqual(cfg["time_col"], "timestamp")
            self.assertEqual(cfg["resample"], "5min")
            self.assertEqual(cfg["agg"], "mean")
            self.assertEqual(cfg["train_split"], 0.7)

    def test_augmentation_parser_supports_seed_and_deterministic(self):
        args = _build_parser().parse_args(["--csv", "toy.csv", "--seed", "123", "--deterministic"])

        self.assertEqual(args.seed, 123)
        self.assertTrue(args.deterministic)

        default_args = _build_parser().parse_args(["--csv", "toy.csv"])
        self.assertIsNone(default_args.seed)
        self.assertFalse(default_args.deterministic)

    def test_augmentation_seed_resolution_prefers_explicit_then_replayed_then_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "run_config.json").write_text(
                json.dumps({"args": {"seed": 77}}),
                encoding="utf-8",
            )

            self.assertEqual(_resolve_augmentation_seed(SimpleNamespace(seed=123), run_dir), 123)
            self.assertEqual(_resolve_augmentation_seed(SimpleNamespace(seed=None), run_dir), 77)
            self.assertEqual(_resolve_augmentation_seed(SimpleNamespace(seed=None), None), 42)

    def test_augmentation_seeded_indices_are_reproducible(self):
        sample_a = _sample_timegan_indices(100, 20, seed=11)
        sample_b = _sample_timegan_indices(100, 20, seed=11)
        sample_c = _sample_timegan_indices(100, 20, seed=12)

        np.testing.assert_array_equal(sample_a, sample_b)
        self.assertFalse(np.array_equal(sample_a, sample_c))

        shuffle_a = _shuffle_mixed_indices(50, seed=11)
        shuffle_b = _shuffle_mixed_indices(50, seed=11)
        shuffle_c = _shuffle_mixed_indices(50, seed=12)

        np.testing.assert_array_equal(shuffle_a, shuffle_b)
        self.assertFalse(np.array_equal(shuffle_a, shuffle_c))

    def test_augmentation_generator_replay_uses_actual_input_width(self):
        generator = build_generator(
            L=12,
            H=3,
            n_in=2,
            units=8,
            num_layers=2,
            dropout=0.0,
            dense_activation="tanh",
            noise_dim=4,
        )
        saved_metrics = {
            "rgan": {
                "config": {
                    "units_g": 8,
                    "g_layers": 2,
                    "noise_dim": 4,
                    "g_dense_activation": "tanh",
                }
            }
        }

        replay_cfg = _infer_generator_replay_config(
            saved_metrics,
            generator.state_dict(),
            n_in_current=2,
            model_label="RGAN",
        )

        self.assertEqual(replay_cfg["n_in"], 2)
        self.assertEqual(replay_cfg["units"], 8)
        self.assertEqual(replay_cfg["num_layers"], 2)
        self.assertEqual(replay_cfg["noise_dim"], 4)
        self.assertEqual(replay_cfg["dense_activation"], "tanh")

        with self.assertRaises(ValueError):
            _infer_generator_replay_config(
                saved_metrics,
                generator.state_dict(),
                n_in_current=1,
                model_label="RGAN",
            )

    def test_detect_rgan_pipeline_prefers_metrics_then_run_config_then_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            models_dir = run_dir / "models"
            models_dir.mkdir()

            self.assertEqual(
                _detect_rgan_pipeline(
                    {"rgan": {"config": {"pipeline": "two_stage"}}},
                    {"pipeline": "joint"},
                    models_dir,
                ),
                "two_stage",
            )
            self.assertEqual(_detect_rgan_pipeline(None, {"pipeline": "two_stage"}, models_dir), "two_stage")

            (models_dir / "rgan_regression.pt").write_bytes(b"x")
            (models_dir / "rgan_residual_generator.pt").write_bytes(b"y")

            self.assertEqual(_detect_rgan_pipeline(None, {}, models_dir), "two_stage")
            self.assertEqual(_detect_rgan_pipeline(None, {}, None), "joint")

    def test_two_stage_replay_config_uses_saved_metadata(self):
        saved_metrics = {
            "rgan": {
                "config": {
                    "pipeline": "two_stage",
                    "n_in": 2,
                    "units_g": 8,
                    "g_layers": 2,
                    "noise_dim": 4,
                }
            }
        }

        replay_cfg = _infer_two_stage_replay_config(saved_metrics, n_in_current=2, model_label="RGAN")

        self.assertEqual(replay_cfg["pipeline"], "two_stage")
        self.assertEqual(replay_cfg["n_in"], 2)
        self.assertEqual(replay_cfg["units"], 8)
        self.assertEqual(replay_cfg["num_layers"], 2)
        self.assertEqual(replay_cfg["noise_dim"], 4)

        with self.assertRaises(ValueError):
            _infer_two_stage_replay_config(saved_metrics, n_in_current=1, model_label="RGAN")

    def test_compute_hybrid_metrics_distinguishes_deterministic_and_stochastic_modes(self):
        class ZeroRegression(torch.nn.Module):
            def __init__(self, horizon):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.horizon = horizon

            def forward(self, x):
                return torch.zeros(x.size(0), self.horizon, 1, device=x.device, dtype=x.dtype)

        class ConstantResidual(torch.nn.Module):
            def __init__(self, horizon, value=1.0):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.noise_dim = 2
                self.H = horizon
                self.value = float(value)

            def forward(self, z):
                return torch.full((z.size(0), self.H, 1), self.value, device=z.device, dtype=z.dtype)

        X = np.zeros((4, 5, 1), dtype=np.float32)
        Y = np.zeros((4, 3, 1), dtype=np.float32)
        regression = ZeroRegression(horizon=3)
        residual = ConstantResidual(horizon=3, value=1.5)

        det_stats, det_pred = compute_hybrid_metrics(regression, residual, X, Y, deterministic=True)
        stoch_stats, stoch_pred = compute_hybrid_metrics(regression, residual, X, Y, deterministic=False)

        np.testing.assert_allclose(det_pred, 0.0)
        np.testing.assert_allclose(stoch_pred, 1.5)
        self.assertLess(det_stats["rmse"], stoch_stats["rmse"])

    def test_predict_rgan_bundle_uses_hybrid_path_for_two_stage(self):
        class ZeroRegression(torch.nn.Module):
            def __init__(self, horizon):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.horizon = horizon

            def forward(self, x):
                return torch.zeros(x.size(0), self.horizon, 1, device=x.device, dtype=x.dtype)

        class ConstantResidual(torch.nn.Module):
            def __init__(self, horizon, value):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.noise_dim = 2
                self.H = horizon
                self.value = float(value)

            def forward(self, z):
                return torch.full((z.size(0), self.H, 1), self.value, device=z.device, dtype=z.dtype)

        X = np.zeros((3, 6, 1), dtype=np.float32)
        Y = np.zeros((3, 2, 1), dtype=np.float32)
        rgan_out = {
            "pipeline": "two_stage",
            "F_hat": ZeroRegression(horizon=2),
            "G": ConstantResidual(horizon=2, value=2.0),
            "G_ema": None,
        }

        det_stats, det_pred = _predict_rgan_bundle(rgan_out, X, Y=Y, deterministic=True, seed=7)
        _, stoch_pred = _predict_rgan_bundle(rgan_out, X, Y=Y, deterministic=False, seed=7)

        np.testing.assert_allclose(det_pred, 0.0)
        np.testing.assert_allclose(stoch_pred, 2.0)
        self.assertEqual(det_stats["rmse"], 0.0)

    def test_load_rgan_bundle_from_run_dir_uses_two_stage_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            models_dir = run_dir / "models"
            models_dir.mkdir()

            metrics = {
                "L": 12,
                "H": 3,
                "rgan": {
                    "config": {
                        "pipeline": "two_stage",
                        "n_in": 1,
                        "units_g": 8,
                        "units_d": 8,
                        "g_layers": 1,
                        "d_layers": 1,
                        "dropout": 0.0,
                        "noise_dim": 4,
                        "d_activation": "linear",
                        "critic_arch": "tcn",
                    }
                },
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run_dir / "run_config.json").write_text(json.dumps({"args": {"pipeline": "two_stage"}}), encoding="utf-8")

            regression = build_regression_model(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
            residual = build_residual_generator(H=3, noise_dim=4, units=8, num_layers=1, dropout=0.0)
            critic = build_residual_discriminator(
                H=3, units=8, num_layers=1, dropout=0.0, activation="linear", critic_arch="tcn"
            )
            torch.save(regression.state_dict(), models_dir / "rgan_regression.pt")
            torch.save(residual.state_dict(), models_dir / "rgan_residual_generator.pt")
            torch.save(critic.state_dict(), models_dir / "rgan_residual_discriminator.pt")
            torch.save(residual.state_dict(), models_dir / "rgan_generator.pt")
            torch.save(critic.state_dict(), models_dir / "rgan_discriminator.pt")

            with mock.patch("rgan.models_torch.build_generator", side_effect=AssertionError("legacy generator should not be built")):
                bundle = _load_rgan_bundle_from_run_dir(
                    run_dir=run_dir,
                    n_in_current=1,
                    L=12,
                    H=3,
                    device=torch.device("cpu"),
                    prefer_ema=False,
                )

            self.assertEqual(bundle["pipeline"], "two_stage")
            self.assertIn("F_hat", bundle)
            self.assertEqual(type(bundle["G"]).__name__, "ResidualGenerator")

    def test_generate_rgan_synthetic_targets_supports_two_stage_hybrid_forecasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            models_dir = run_dir / "models"
            models_dir.mkdir()

            metrics = {
                "L": 12,
                "H": 3,
                "rgan": {
                    "config": {
                        "pipeline": "two_stage",
                        "n_in": 1,
                        "units_g": 8,
                        "units_d": 8,
                        "g_layers": 1,
                        "d_layers": 1,
                        "dropout": 0.0,
                        "noise_dim": 4,
                        "d_activation": "linear",
                        "critic_arch": "tcn",
                    }
                },
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run_dir / "run_config.json").write_text(json.dumps({"args": {"pipeline": "two_stage"}}), encoding="utf-8")

            regression = build_regression_model(L=12, H=3, n_in=1, units=8, num_layers=1, dropout=0.0)
            residual = build_residual_generator(H=3, noise_dim=4, units=8, num_layers=1, dropout=0.0)
            critic = build_residual_discriminator(
                H=3, units=8, num_layers=1, dropout=0.0, activation="linear", critic_arch="tcn"
            )
            torch.save(regression.state_dict(), models_dir / "rgan_regression.pt")
            torch.save(residual.state_dict(), models_dir / "rgan_residual_generator.pt")
            torch.save(critic.state_dict(), models_dir / "rgan_residual_discriminator.pt")
            torch.save(residual.state_dict(), models_dir / "rgan_generator.pt")
            torch.save(critic.state_dict(), models_dir / "rgan_discriminator.pt")

            bundle = _load_rgan_bundle_from_run_dir(
                run_dir=run_dir,
                n_in_current=1,
                L=12,
                H=3,
                device=torch.device("cpu"),
                prefer_ema=False,
            )
            X = np.zeros((5, 12, 1), dtype=np.float32)
            Y = _generate_rgan_synthetic_targets(bundle, X, seed=13, n_runs=2, batch_size=2)

            self.assertEqual(Y.shape, (5, 3, 1))

    def test_covariate_windowing_preserves_input_width_for_augmentation(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=40, freq="h"),
                "target": np.linspace(0.0, 1.0, 40),
                "covar": np.linspace(1.0, 2.0, 40),
            }
        )
        prep = interpolate_and_standardize(df, "target", train_ratio=0.8)
        X, Y = make_windows_with_covariates(prep["scaled_train"], "target", prep["covariates"], 8, 2)

        self.assertEqual(X.shape[-1], 2)
        self.assertEqual(Y.shape[-1], 1)

    def test_augmentation_classifiers_are_seed_stable(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(48, 8, 1)).astype(np.float32)
        y_train = (X_train[:, -1, 0] > 0).astype(int)
        X_test = rng.normal(size=(16, 8, 1)).astype(np.float32)
        y_test = (X_test[:, -1, 0] > 0).astype(int)

        results_a = train_evaluate_classifiers(X_train, y_train, X_test, y_test, seed=19)
        results_b = train_evaluate_classifiers(X_train, y_train, X_test, y_test, seed=19)

        self.assertEqual(results_a, results_b)

    def test_evaluate_discriminators_accepts_seeded_reproducible_classifiers(self):
        rng = np.random.default_rng(3)
        real = rng.normal(size=(24, 6, 1)).astype(np.float32)
        fake = (rng.normal(size=(24, 6, 1)) + 0.5).astype(np.float32)

        results_a = evaluate_discriminators(real, fake, n_folds=3, seed=23)
        results_b = evaluate_discriminators(real, fake, n_folds=3, seed=23)

        self.assertEqual(results_a, results_b)

    def test_train_timegan_is_reproducible_with_explicit_seed(self):
        real_data = np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(8, 4, 1)

        out_a = train_timegan(
            real_data,
            hidden_dim=4,
            latent_dim=4,
            n_layers=1,
            epochs_ae=1,
            epochs_sup=1,
            epochs_joint=1,
            batch_size=2,
            lr=1e-3,
            device="cpu",
            seed=29,
        )
        out_b = train_timegan(
            real_data,
            hidden_dim=4,
            latent_dim=4,
            n_layers=1,
            epochs_ae=1,
            epochs_sup=1,
            epochs_joint=1,
            batch_size=2,
            lr=1e-3,
            device="cpu",
            seed=29,
        )

        np.testing.assert_allclose(out_a["synthetic_data"], out_b["synthetic_data"])

    def test_benchmark_invalid_gpu_id_falls_back_to_cpu(self):
        with mock.patch("rgan.scripts.run_benchmark.torch.cuda.is_available", return_value=True), \
             mock.patch("rgan.scripts.run_benchmark.torch.cuda.device_count", return_value=1):
            self.assertEqual(_device_str(99), "cpu")

    def test_build_paper_accepts_template_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            metrics_path = tmpdir / "metrics.json"
            template_path = tmpdir / "template.tex"
            out_path = tmpdir / "paper.tex"

            metrics_path.write_text(json.dumps({
                "charts": {
                    "training_curves_overlay": "",
                    "ranked_model_bars": "",
                },
                "rgan": {
                    "architecture": {
                        "generator": ["LSTM(64)"],
                        "discriminator": ["LSTM(64)"],
                    },
                    "config": {
                        "g_layers": 1,
                        "units_g": 64,
                        "g_dense": "linear",
                        "d_layers": 1,
                        "units_d": 64,
                        "d_activation": "sigmoid",
                        "lambda_reg": 1.0,
                        "dropout": 0.1,
                        "lrG": 0.001,
                        "lrD": 0.001,
                    },
                    "train": {"rmse": 1.0, "mse": 1.0, "bias": 0.0},
                    "test": {"rmse": 1.1, "mse": 1.21, "bias": 0.0},
                },
            }), encoding="utf-8")
            template_path.write_text(
                "\n".join([
                    "%(learning_curve)s",
                    "%(compare_test)s",
                    "%(rgan_curve)s",
                    "%(generator_arch)s",
                    "%(discriminator_arch)s",
                    "%(rgan_hparams)s",
                    "%(error_table_rows)s",
                ]),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rgan.scripts.build_paper",
                    "--metrics",
                    str(metrics_path),
                    "--template",
                    str(template_path),
                    "--out",
                    str(out_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(out_path.exists())

    def test_run_training_supports_rgan_only_without_prior_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            csv_path = tmpdir / "toy.csv"
            results_dir = tmpdir / "results"

            rows = ["timestamp,value"]
            start_ms = 1704067200000
            for i in range(160):
                rows.append(f"{start_ms + (i * 60000)},{1.0 + (i * 0.01):.6f}")
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rgan.scripts.run_training",
                    "--csv",
                    str(csv_path),
                    "--target",
                    "value",
                    "--time_col",
                    "timestamp",
                    "--L",
                    "12",
                    "--H",
                    "3",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "8",
                    "--eval_batch_size",
                    "16",
                    "--noise_levels",
                    "0",
                    "--bootstrap_samples",
                    "0",
                    "--skip_classical",
                    "--skip_noise_robustness",
                    "--only_models",
                    "rgan",
                    "--critic_arch",
                    "tcn",
                    "--preload_to_device",
                    "never",
                    "--compile_mode",
                    "off",
                    "--results_dir",
                    str(results_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("rgan", metrics)
            self.assertEqual(metrics["rgan"]["config"]["critic_arch"], "tcn")
            self.assertTrue(metrics["lstm"].get("skipped", False))
            self.assertTrue(metrics["dlinear"].get("skipped", False))

    def test_run_training_supports_two_stage_rgan_only_and_writes_explicit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            csv_path = tmpdir / "toy.csv"
            results_dir = tmpdir / "results"

            rows = ["timestamp,value"]
            start_ms = 1704067200000
            for i in range(180):
                rows.append(f"{start_ms + (i * 60000)},{1.0 + (i * 0.01):.6f}")
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rgan.scripts.run_training",
                    "--csv",
                    str(csv_path),
                    "--target",
                    "value",
                    "--time_col",
                    "timestamp",
                    "--L",
                    "12",
                    "--H",
                    "3",
                    "--epochs",
                    "1",
                    "--regression_epochs",
                    "1",
                    "--pipeline",
                    "two_stage",
                    "--batch_size",
                    "8",
                    "--eval_batch_size",
                    "16",
                    "--noise_levels",
                    "0",
                    "--bootstrap_samples",
                    "0",
                    "--skip_classical",
                    "--skip_noise_robustness",
                    "--only_models",
                    "rgan",
                    "--critic_arch",
                    "tcn",
                    "--preload_to_device",
                    "never",
                    "--compile_mode",
                    "off",
                    "--results_dir",
                    str(results_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            saved_models = {path.name for path in (results_dir / "models").glob("*.pt")}
            self.assertIn("rgan_regression.pt", saved_models)
            self.assertIn("rgan_residual_generator.pt", saved_models)
            self.assertIn("rgan_residual_discriminator.pt", saved_models)
            metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["rgan"]["config"]["pipeline"], "two_stage")

    def test_selective_retrain_saves_loaded_models_into_new_results_dir(self):
        from rgan.autoformer import Autoformer
        from rgan.fits import FITS
        from rgan.informer import Informer
        from rgan.itransformer import iTransformer
        from rgan.linear_baselines import DLinear, NLinear
        from rgan.lstm_supervised_torch import LSTMSupervised
        from rgan.patchtst import PatchTST

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            csv_path = tmpdir / "toy.csv"
            prior_dir = tmpdir / "prior"
            prior_models = prior_dir / "models"
            results_dir = tmpdir / "results"
            prior_models.mkdir(parents=True)

            rows = ["timestamp,value"]
            start_ms = 1704067200000
            for i in range(220):
                rows.append(f"{start_ms + (i * 60000)},{1.0 + (i * 0.01):.6f}")
            csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            model_defs = {
                "lstm.pt": LSTMSupervised(L=24, H=6, n_in=1, units=8),
                "dlinear.pt": DLinear(L=24, H=6),
                "nlinear.pt": NLinear(L=24, H=6),
                "fits.pt": FITS(L=24, H=6),
                "patchtst.pt": PatchTST(L=24, H=6),
                "itransformer.pt": iTransformer(L=24, H=6),
                "autoformer.pt": Autoformer(L=24, H=6),
                "informer.pt": Informer(L=24, H=6),
            }
            for filename, model in model_defs.items():
                torch.save(model.state_dict(), prior_models / filename)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rgan.scripts.run_training",
                    "--csv",
                    str(csv_path),
                    "--target",
                    "value",
                    "--time_col",
                    "timestamp",
                    "--L",
                    "24",
                    "--H",
                    "6",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "8",
                    "--eval_batch_size",
                    "16",
                    "--units_g",
                    "8",
                    "--units_d",
                    "8",
                    "--g_layers",
                    "1",
                    "--d_layers",
                    "1",
                    "--noise_levels",
                    "0",
                    "--bootstrap_samples",
                    "0",
                    "--skip_classical",
                    "--skip_noise_robustness",
                    "--only_models",
                    "rgan",
                    "--prior_results",
                    str(prior_dir),
                    "--critic_arch",
                    "tcn",
                    "--preload_to_device",
                    "never",
                    "--compile_mode",
                    "off",
                    "--results_dir",
                    str(results_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            saved_models = {path.name for path in (results_dir / "models").glob("*.pt")}
            self.assertIn("rgan_generator.pt", saved_models)
            self.assertIn("rgan_discriminator.pt", saved_models)
            for filename in model_defs:
                self.assertIn(filename, saved_models)

    def test_training_curve_overlay_renders_with_validation_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = plot_training_curves_overlay(
                model_histories={
                    "RGAN": {
                        "epoch": [5, 10, 15],
                        "val_rmse": [0.5, 0.4, 0.35],
                        "test_rmse": [float("nan"), float("nan"), float("nan")],
                    }
                },
                classical_baselines={},
                out_path=str(Path(tmp) / "overlay"),
                metric="val_rmse",
                ylabel="Validation RMSE",
            )

            self.assertTrue(Path(out["static"]).exists())

    def test_benchmark_exits_nonzero_when_every_run_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            results_dir = Path(tmp) / "results"
            data_dir.mkdir()

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rgan.scripts.run_benchmark",
                    "--datasets",
                    "ETTh1",
                    "--pred_lens",
                    "96",
                    "--epochs",
                    "1",
                    "--batch_size",
                    "16",
                    "--skip_classical",
                    "--data_dir",
                    str(data_dir),
                    "--results_dir",
                    str(results_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No results collected.", result.stdout)


if __name__ == "__main__":
    unittest.main()
