'use strict';

(function () {
  const { useState, useEffect, useMemo, useRef } = React;

  const MODEL_META = [
    { key: 'rgan', label: 'R-GAN', color: '#6366F1' },
    { key: 'lstm', label: 'LSTM', color: '#EC4899' },
    { key: 'naive_baseline', label: 'Naïve Baseline', color: '#22D3EE' },
    { key: 'naive_bayes', label: 'Naïve Bayes', color: '#10B981' },
  ];

  const COLOR_SCALE = MODEL_META.reduce((acc, item) => {
    acc[item.key] = item.color;
    return acc;
  }, {});

  const CONFIG_LABELS = {
    units_g: 'Generator units',
    units_d: 'Discriminator units',
    lambda_reg: 'λ regularization',
    lrG: 'Generator learning rate',
    lrD: 'Discriminator learning rate',
    dropout: 'Dropout',
    g_layers: 'Generator layers',
    d_layers: 'Discriminator layers',
    g_dense: 'Generator dense activation',
    g_dense_activation: 'Generator dense activation',
    d_activation: 'Discriminator activation',
    units: 'Hidden units',
    lr: 'Learning rate',
    batch_size: 'Batch size',
    patience: 'Early stopping patience',
  };

  const DEFAULT_METRICS_PATH = (function () {
    const params = new URLSearchParams(window.location.search);
    return params.get('metrics') || '../results/metrics.json';
  })();

  const formatNumber = (value, decimals = 4) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '–';
    }
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
  };

  const formatPercent = (value, decimals = 1) => {
    if (value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value)) {
      return '–';
    }
    const percent = value * 100;
    const rounded = formatNumber(percent, decimals);
    const sign = percent > 0 ? '+' : '';
    return `${sign}${rounded}%`;
  };

  const labelize = (key) => {
    if (!key) return '';
    if (CONFIG_LABELS[key]) return CONFIG_LABELS[key];
    return key
      .replace(/[_-]+/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const formatConfigValue = (value) => {
    if (value === null || value === undefined || value === '') return '—';
    if (typeof value === 'number') {
      if (Number.isInteger(value)) {
        return value.toString();
      }
      return formatNumber(value, 6);
    }
    if (typeof value === 'boolean') {
      return value ? 'Enabled' : 'Disabled';
    }
    if (Array.isArray(value)) {
      return value.map((item) => formatConfigValue(item)).join(', ');
    }
    return String(value);
  };

  const formatConfidence = (stats, key) => {
    if (!stats) return '–';
    const low = stats[`${key}_orig_ci_low`] ?? stats[`${key}_ci_low`];
    const high = stats[`${key}_orig_ci_high`] ?? stats[`${key}_ci_high`];
    if (low === undefined || high === undefined) {
      const std = stats[`${key}_orig_std`] ?? stats[`${key}_std`];
      return std === undefined ? '–' : `±${formatNumber(std)}`;
    }
    return `[${formatNumber(low)}, ${formatNumber(high)}]`;
  };

  const PlotlyChart = ({ id, data, layout, config }) => {
    const containerRef = useRef(null);

    useEffect(() => {
      if (!containerRef.current || !Array.isArray(data)) {
        return undefined;
      }
      const plotLayout = Object.assign({ autosize: true, margin: { t: 48, r: 24, b: 48, l: 56 } }, layout || {});
      const plotConfig = Object.assign({ responsive: true, displaylogo: false }, config || {});

      Plotly.react(containerRef.current, data, plotLayout, plotConfig);
      return () => {
        Plotly.purge(containerRef.current);
      };
    }, [containerRef, data, layout, config, id]);

    return React.createElement('div', { id, ref: containerRef, className: 'plotly-chart' });
  };

  const SummaryCards = ({ metrics }) => {
    const cards = useMemo(() => {
      if (!metrics) return [];
      return MODEL_META.map((model) => {
        const detail = metrics[model.key];
        if (!detail) return null;
        const train = detail.train || {};
        const test = detail.test || {};
        return {
          key: model.key,
          title: model.label,
          color: model.color,
          trainRmse: train.rmse_orig ?? train.rmse ?? null,
          testRmse: test.rmse_orig ?? test.rmse ?? null,
          testMae: test.mae_orig ?? test.mae ?? null,
          noise:
            metrics.noise_robustness?.[0]?.[model.key]?.rmse_orig ??
            metrics.noise_robustness?.[0]?.[model.key]?.rmse ??
            null,
        };
      }).filter(Boolean);
    }, [metrics]);

    return React.createElement(
      'section',
      { className: 'section summary-section' },
      React.createElement('header', { className: 'section-header' }, [
        React.createElement('h2', { key: 'title' }, 'Model Snapshot'),
        React.createElement(
          'p',
          { key: 'subtitle', className: 'section-subtitle' },
          'Key performance indicators in original data units when available.'
        ),
      ]),
      React.createElement(
        'div',
        { className: 'summary-grid' },
        cards.map((card) =>
          React.createElement(
            'article',
            { key: card.key, className: 'summary-card', style: { borderTopColor: card.color } },
            [
              React.createElement('h3', { key: 'title' }, card.title),
              React.createElement(
                'div',
                { key: 'metrics', className: 'summary-metrics' },
                [
                  React.createElement(
                    'div',
                    { key: 'test' },
                    [
                      React.createElement('span', { key: 'label', className: 'metric-label' }, 'Test RMSE'),
                      React.createElement('span', { key: 'value', className: 'metric-value' }, formatNumber(card.testRmse)),
                    ]
                  ),
                  React.createElement(
                    'div',
                    { key: 'train' },
                    [
                      React.createElement('span', { key: 'label', className: 'metric-label' }, 'Train RMSE'),
                      React.createElement('span', { key: 'value', className: 'metric-value' }, formatNumber(card.trainRmse)),
                    ]
                  ),
                  React.createElement(
                    'div',
                    { key: 'mae' },
                    [
                      React.createElement('span', { key: 'label', className: 'metric-label' }, 'Test MAE'),
                      React.createElement('span', { key: 'value', className: 'metric-value' }, formatNumber(card.testMae)),
                    ]
                  ),
                ]
              ),
              card.noise !== null
                ? React.createElement(
                    'footer',
                    { key: 'footer', className: 'summary-footnote' },
                    `Noise-free RMSE: ${formatNumber(card.noise)}`
                  )
                : null,
            ]
          )
        )
      )
    );
  };

  const DatasetOverview = ({ metrics }) => {
    if (!metrics) return null;
    const dataset = metrics.dataset || 'Unknown dataset';
    const tuning = metrics.tuning_dataset || metrics.tuning?.dataset || '—';
    const env = metrics.environment || {};
    const scaling = metrics.scaling || {};

    return React.createElement(
      'section',
      { className: 'section overview-section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Experiment Overview'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Data provenance, reproducibility settings, and environment metadata.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'content', className: 'overview-grid' },
          [
            React.createElement(
              'article',
              { key: 'dataset', className: 'card' },
              [
                React.createElement('h3', { key: 'title' }, 'Datasets'),
                React.createElement(
                  'ul',
                  { key: 'list', className: 'card-list' },
                  [
                    ['Primary dataset', dataset],
                    ['Tuning dataset', tuning],
                    ['Target column', metrics.target_col || 'auto'],
                    ['Time column', metrics.time_col_used || '(none)'],
                    [
                      'Train/Test windows',
                      `${metrics.num_train_windows || 0} / ${metrics.num_test_windows || 0}`,
                    ],
                  ].map(([label, value]) =>
                    React.createElement(
                      'li',
                      { key: label },
                      [
                        React.createElement('span', { key: 'label', className: 'item-label' }, label),
                        React.createElement('span', { key: 'value', className: 'item-value' }, value),
                      ]
                    )
                  )
                ),
              ]
            ),
            React.createElement(
              'article',
              { key: 'repro', className: 'card' },
              [
                React.createElement('h3', { key: 'title' }, 'Reproducibility'),
                React.createElement(
                  'ul',
                  { key: 'list', className: 'card-list' },
                  [
                    ['Seed', metrics.seed ?? env.seed ?? '—'],
                    ['Scaling mean', formatNumber(scaling.target_mean ?? NaN)],
                    ['Scaling std', formatNumber(scaling.target_std ?? NaN)],
                    ['Created (UTC)', metrics.created || '—'],
                    ['Git commit', env.git_commit ? env.git_commit.slice(0, 10) : '—'],
                  ].map(([label, value]) =>
                    React.createElement(
                      'li',
                      { key: label },
                      [
                        React.createElement('span', { key: 'label', className: 'item-label' }, label),
                        React.createElement('span', { key: 'value', className: 'item-value monospace' }, value),
                      ]
                    )
                  )
                ),
              ]
            ),
            React.createElement(
              'article',
              { key: 'environment', className: 'card' },
              [
                React.createElement('h3', { key: 'title' }, 'Environment'),
                React.createElement(
                  'ul',
                  { key: 'list', className: 'card-list' },
                  [
                    ['Python', env.python || '—'],
                    ['Platform', env.platform || '—'],
                    [
                      'Packages',
                      (env.packages &&
                        Object.entries(env.packages)
                          .map(([pkg, version]) => `${pkg}@${version}`)
                          .join(', ')) ||
                        '—',
                    ],
                  ].map(([label, value]) =>
                    React.createElement(
                      'li',
                      { key: label },
                      [
                        React.createElement('span', { key: 'label', className: 'item-label' }, label),
                        React.createElement('span', { key: 'value', className: 'item-value' }, value),
                      ]
                    )
                  )
                ),
              ]
            ),
          ]
        ),
      ]
    );
  };

  const HighlightsGrid = ({ metrics }) => {
    const callouts = useMemo(() => {
      if (!metrics) return [];
      const summaries = MODEL_META.map((model) => {
        const detail = metrics[model.key];
        if (!detail) return null;
        const train = detail.train || {};
        const test = detail.test || {};
        const trainRmse = train.rmse_orig ?? train.rmse;
        const testRmse = test.rmse_orig ?? test.rmse;
        if (testRmse === undefined || testRmse === null) return null;
        return {
          key: model.key,
          label: model.label,
          color: model.color,
          trainRmse,
          testRmse,
        };
      }).filter(Boolean);
      if (!summaries.length) return [];

      const best = summaries.reduce((acc, item) => {
        if (item.testRmse === null || item.testRmse === undefined) return acc;
        if (!acc || item.testRmse < acc.testRmse) return item;
        return acc;
      }, null);

      const naive = summaries.find((item) => item.key === 'naive_baseline');
      const bestImprovement = best && naive && naive.testRmse
        ? (naive.testRmse - best.testRmse) / naive.testRmse
        : null;

      const gaps = summaries
        .map((item) => {
          if (item.trainRmse === undefined || item.trainRmse === null) return null;
          return {
            ...item,
            gap: item.testRmse - item.trainRmse,
          };
        })
        .filter(Boolean);
      const mostStable = gaps.reduce((acc, item) => {
        if (!acc) return item;
        return Math.abs(item.gap) < Math.abs(acc.gap) ? item : acc;
      }, null);

      const noiseEntries = Array.isArray(metrics.noise_robustness) ? metrics.noise_robustness : [];
      let noiseSummary = null;
      if (best && noiseEntries.length >= 2) {
        const base = noiseEntries.find((entry) => entry.sd === 0) || noiseEntries[0];
        const extreme = noiseEntries[noiseEntries.length - 1];
        const baseStats = base?.[best.key] || {};
        const extremeStats = extreme?.[best.key] || {};
        const baseRmse = baseStats.rmse_orig ?? baseStats.rmse;
        const extremeRmse = extremeStats.rmse_orig ?? extremeStats.rmse;
        if (baseRmse !== null && baseRmse !== undefined && extremeRmse !== null && extremeRmse !== undefined) {
          noiseSummary = {
            sd: extreme.sd,
            delta: extremeRmse - baseRmse,
            percent: baseRmse ? (extremeRmse - baseRmse) / baseRmse : null,
          };
        }
      }

      const classical = metrics.classical || {};
      const classicalOptions = [
        { label: 'ETS', value: classical.ets_rmse_full },
        { label: 'ARIMA', value: classical.arima_rmse_full },
      ].filter((item) => item.value !== null && item.value !== undefined && !Number.isNaN(item.value));
      const bestClassical = classicalOptions.reduce((acc, item) => {
        if (!acc) return item;
        return item.value < acc.value ? item : acc;
      }, null);

      const cards = [];
      if (best) {
        cards.push({
          title: 'Top performer',
          metric: `${best.label}`,
          value: `Test RMSE ${formatNumber(best.testRmse)}`,
          detail:
            bestImprovement !== null
              ? `${formatPercent(bestImprovement)} vs naïve baseline`
              : 'Lowest test RMSE across evaluated models.',
          accent: best.color,
        });
      }
      if (mostStable) {
        cards.push({
          title: 'Generalization gap',
          metric: `${formatNumber(mostStable.gap)}`,
          value: mostStable.label,
          detail: 'Smallest |train − test| RMSE difference.',
          accent: mostStable.color,
        });
      }
      if (noiseSummary && best) {
        cards.push({
          title: 'Noise resilience',
          metric: `${formatNumber(noiseSummary.delta)}`,
          value: `ΔRMSE @ σ=${formatNumber(noiseSummary.sd, 2)}`,
          detail:
            noiseSummary.percent !== null
              ? `${formatPercent(noiseSummary.percent)} shift for ${best.label}`
              : `RMSE delta for ${best.label}`,
          accent: '#F97316',
        });
      }
      if (bestClassical) {
        cards.push({
          title: 'Classical baseline',
          metric: `${bestClassical.label}`,
          value: `RMSE ${formatNumber(bestClassical.value)}`,
          detail: 'Best-performing statistical reference model.',
          accent: '#F59E0B',
        });
      }
      return cards;
    }, [metrics]);

    if (!callouts.length) return null;

    return React.createElement(
      'section',
      { className: 'section callout-section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Executive Highlights'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Key takeaways across performance, generalization, and robustness metrics.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'grid', className: 'callout-grid' },
          callouts.map((card) =>
            React.createElement(
              'article',
              {
                key: card.title,
                className: 'callout-card',
                style: { borderTopColor: card.accent },
              },
              [
                React.createElement('span', { key: 'title', className: 'callout-title' }, card.title),
                React.createElement('strong', { key: 'metric', className: 'callout-metric' }, card.metric),
                React.createElement('span', { key: 'value', className: 'callout-value' }, card.value),
                React.createElement('p', { key: 'detail', className: 'callout-detail' }, card.detail),
              ]
            )
          )
        ),
      ]
    );
  };

  const NoiseRobustnessChart = ({ metrics }) => {
    const data = useMemo(() => {
      if (!metrics?.noise_robustness) return null;
      const sdLevels = metrics.noise_robustness.map((entry) => entry.sd);
      return MODEL_META.map((model) => {
        const series = metrics.noise_robustness.map((entry) => {
          const stats = entry[model.key] || {};
          return stats.rmse_orig ?? stats.rmse ?? null;
        });
        return {
          x: sdLevels,
          y: series,
          mode: 'lines+markers',
          name: model.label,
          line: { color: model.color, width: 3 },
          marker: { size: 8 },
          hovertemplate: 'Noise σ=%{x}<br>RMSE=%{y:.6f}<extra>' + model.label + '</extra>',
        };
      }).filter(Boolean);
    }, [metrics]);

    if (!data || data.length === 0) return null;

    const layout = {
      title: 'Noise Robustness (RMSE vs. Gaussian perturbation)',
      xaxis: { title: 'Noise standard deviation (σ)' },
      yaxis: { title: 'RMSE', rangemode: 'tozero' },
      legend: { orientation: 'h', y: -0.2 },
    };

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Noise Robustness'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Performance degradation under additive Gaussian noise injected into the test window inputs.'
          ),
        ]),
        React.createElement(PlotlyChart, { key: 'chart', id: 'noise-chart', data, layout }),
      ]
    );
  };

  const LearningCurveChart = ({ metrics }) => {
    const { data, layout, available } = useMemo(() => {
      const result = { data: [], layout: null, available: false };
      const info = metrics?.learning_curves;
      if (!info || !Array.isArray(info.sizes) || info.sizes.length === 0) {
        return result;
      }
      const sizes = info.sizes;
      const means = info.curves?.means || {};
      const stds = info.curves?.stds || {};
      const traces = MODEL_META.map((model) => {
        const series = means[model.label] || means[model.key] || null;
        if (!series) return null;
        const sigma = stds[model.label] || stds[model.key] || [];
        return {
          x: sizes,
          y: series,
          mode: 'lines+markers',
          name: model.label,
          line: { color: model.color, width: 3 },
          marker: { size: 8 },
          hovertemplate: 'Windows=%{x}<br>RMSE=%{y:.6f}<extra>' + model.label + '</extra>',
          error_y: sigma.length
            ? {
                type: 'data',
                array: sigma,
                visible: true,
                color: model.color,
              }
            : undefined,
        };
      }).filter(Boolean);
      if (traces.length === 0) {
        return result;
      }
      result.data = traces;
      result.layout = {
        title: 'Learning Curves (RMSE vs. training windows)',
        xaxis: { title: 'Number of training windows', tickmode: 'linear' },
        yaxis: { title: 'RMSE', rangemode: 'tozero' },
        legend: { orientation: 'h', y: -0.2 },
      };
      result.available = true;
      return result;
    }, [metrics]);

    if (!available) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Learning Curves'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Mean RMSE across resampled training set sizes with uncertainty bands (±1σ).'
          ),
        ]),
        React.createElement(PlotlyChart, { key: 'chart', id: 'learning-curve', data, layout }),
      ]
    );
  };

  const TrainingHistorySection = ({ metrics }) => {
    const { traces, layout } = useMemo(() => {
      const traces = [];
      const layout = {
        title: 'Training Dynamics',
        xaxis: { title: 'Epoch', rangemode: 'tozero' },
        yaxis: { title: 'RMSE', rangemode: 'tozero' },
        legend: { orientation: 'h', y: -0.2 },
      };

      const pushHistory = (modelKey, meta) => {
        const history = metrics?.[modelKey]?.history;
        if (!history || !Array.isArray(history.epoch) || !history.epoch.length) {
          return;
        }
        const xValues = history.epoch;
        const trainSeries = history.train_rmse;
        const testSeries = history.test_rmse;
        if (Array.isArray(trainSeries) && trainSeries.length) {
          traces.push({
            x: xValues,
            y: trainSeries,
            mode: 'lines',
            name: `${meta.label} · Train`,
            line: { color: meta.color, width: 3 },
          });
        }
        if (Array.isArray(testSeries) && testSeries.length) {
          traces.push({
            x: xValues,
            y: testSeries,
            mode: 'lines',
            name: `${meta.label} · Test`,
            line: { color: meta.color, width: 3, dash: 'dash' },
          });
        }
      };

      MODEL_META.forEach((meta) => pushHistory(meta.key, meta));

      return { traces, layout };
    }, [metrics]);

    if (!traces.length) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Learning Trajectories'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Epoch-wise RMSE progression for train and test splits across each model.'
          ),
        ]),
        React.createElement(PlotlyChart, { key: 'chart', id: 'history-chart', data: traces, layout }),
      ]
    );
  };

  const ComparisonBarChart = ({ metrics }) => {
    const data = useMemo(() => {
      const testErrors = metrics?.compare_plots ? MODEL_META.map((model) => {
        const stats = metrics[model.key]?.test || {};
        const rmse = stats.rmse_orig ?? stats.rmse;
        return { model, rmse };
      }) : [];
      const trainErrors = metrics?.compare_plots ? MODEL_META.map((model) => {
        const stats = metrics[model.key]?.train || {};
        const rmse = stats.rmse_orig ?? stats.rmse;
        return { model, rmse };
      }) : [];
      const anyData = testErrors.some((item) => item.rmse !== undefined && item.rmse !== null);
      if (!anyData) return null;

      return {
        data: [
          {
            x: testErrors.map((item) => item.model.label),
            y: testErrors.map((item) => item.rmse),
            type: 'bar',
            name: 'Test RMSE',
            marker: { color: testErrors.map((item) => item.model.color) },
            hovertemplate: 'Model=%{x}<br>RMSE=%{y:.6f}<extra>Test</extra>',
          },
          {
            x: trainErrors.map((item) => item.model.label),
            y: trainErrors.map((item) => item.rmse),
            type: 'bar',
            name: 'Train RMSE',
            marker: { color: trainErrors.map((item) => `${item.model.color}99`) },
            hovertemplate: 'Model=%{x}<br>RMSE=%{y:.6f}<extra>Train</extra>',
          },
        ],
        layout: {
          barmode: 'group',
          title: 'Train vs. Test RMSE',
          yaxis: { title: 'RMSE', rangemode: 'tozero' },
          legend: { orientation: 'h', y: -0.2 },
        },
      };
    }, [metrics]);

    if (!data) return null;
    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Model Comparison'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Direct comparison of train/test RMSE performance across baselines and proposed models.'
          ),
        ]),
        React.createElement(PlotlyChart, {
          key: 'chart',
          id: 'comparison-chart',
          data: data.data,
          layout: data.layout,
        }),
      ]
    );
  };

  const PrecisionTable = ({ metrics }) => {
    const rows = useMemo(() => {
      if (!metrics) return [];
      return MODEL_META.flatMap((model) => {
        const detail = metrics[model.key];
        if (!detail) return [];
        return [
          {
            key: `${model.key}-train`,
            label: `${model.label} (Train)`,
            stats: detail.train,
          },
          {
            key: `${model.key}-test`,
            label: `${model.label} (Test)`,
            stats: detail.test,
          },
        ];
      });
    }, [metrics]);

    if (!rows.length) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Metric Breakdown'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'RMSE, MAE, MSE, and Bias with bootstrap-derived 95% confidence intervals when available.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'table-wrapper', className: 'table-wrapper' },
          React.createElement(
            'table',
            { className: 'metrics-table' },
            [
              React.createElement(
                'thead',
                { key: 'head' },
                React.createElement(
                  'tr',
                  null,
                  ['Model', 'RMSE', 'RMSE CI', 'MAE', 'MAE CI', 'MSE', 'Bias'].map((header) =>
                    React.createElement('th', { key: header }, header)
                  )
                )
              ),
              React.createElement(
                'tbody',
                { key: 'body' },
                rows.map((row) =>
                  React.createElement(
                    'tr',
                    { key: row.key },
                    [
                      React.createElement('td', { key: 'model' }, row.label),
                      React.createElement('td', { key: 'rmse' }, formatNumber(row.stats?.rmse_orig ?? row.stats?.rmse)),
                      React.createElement('td', { key: 'rmse_ci' }, formatConfidence(row.stats, 'rmse')),
                      React.createElement('td', { key: 'mae' }, formatNumber(row.stats?.mae_orig ?? row.stats?.mae)),
                      React.createElement('td', { key: 'mae_ci' }, formatConfidence(row.stats, 'mae')),
                      React.createElement('td', { key: 'mse' }, formatNumber(row.stats?.mse_orig ?? row.stats?.mse)),
                      React.createElement('td', { key: 'bias' }, formatNumber(row.stats?.bias_orig ?? row.stats?.bias)),
                    ]
                  )
                )
              ),
            ]
          )
        ),
      ]
    );
  };

  const ConfigurationPanel = ({ metrics }) => {
    const cards = useMemo(() => {
      if (!metrics) return [];
      const entries = [];

      if (metrics.rgan?.config) {
        entries.push({
          key: 'rgan-config',
          title: 'R-GAN Configuration',
          accent: COLOR_SCALE.rgan,
          config: metrics.rgan.config,
        });
      }
      if (metrics.lstm?.config) {
        entries.push({
          key: 'lstm-config',
          title: 'LSTM Baseline Configuration',
          accent: COLOR_SCALE.lstm,
          config: metrics.lstm.config,
        });
      }

      return entries;
    }, [metrics]);

    if (!cards.length) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Configuration & Tuning'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Hyperparameters captured at experiment time and any tuned overrides applied in this run.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'grid', className: 'config-grid' },
          cards.map((card) =>
            React.createElement(
              'article',
              {
                key: card.key,
                className: 'config-card',
                style: { borderTopColor: card.accent },
              },
              [
                React.createElement('h3', { key: 'title' }, card.title),
                React.createElement(
                  'ul',
                  { key: 'list', className: 'config-list' },
                  Object.entries(card.config || {}).map(([key, value]) =>
                    React.createElement(
                      'li',
                      { key },
                      [
                        React.createElement('span', { key: 'label', className: 'config-label' }, labelize(key)),
                        React.createElement('span', { key: 'value', className: 'config-value' }, formatConfigValue(value)),
                      ]
                    )
                  )
                ),
              ]
            )
          )
        ),
      ]
    );
  };

  const TuningSummary = ({ metrics }) => {
    const tuning = metrics?.tuning;
    const enabled = Boolean(tuning?.enabled);
    const dataset = tuning?.dataset || metrics?.tuning_dataset || '—';
    const best = tuning?.best && typeof tuning.best === 'object' ? tuning.best : {};
    const highlightKeys = Object.entries(best).filter(([key]) => !['seed'].includes(key));
    const hasContent = enabled || highlightKeys.length > 0;

    if (!hasContent) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Hyperparameter Tuning'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Summary of the sweep configuration and the winning parameter set when tuning was enabled.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'content', className: 'tuning-card' },
          [
            React.createElement(
              'div',
              { key: 'status-line', className: 'tuning-header' },
              [
                React.createElement(
                  'span',
                  { key: 'status', className: `status-pill ${enabled ? 'status-ready' : 'status-muted'}` },
                  enabled ? 'Sweep executed' : 'Sweep skipped'
                ),
                React.createElement(
                  'span',
                  { key: 'dataset', className: 'tuning-dataset' },
                  `Dataset: ${dataset}`
                ),
              ]
            ),
            highlightKeys.length
              ? React.createElement(
                  'ul',
                  { key: 'list', className: 'config-list tuning-list' },
                  highlightKeys.map(([key, value]) =>
                    React.createElement(
                      'li',
                      { key },
                      [
                        React.createElement('span', { key: 'label', className: 'config-label' }, labelize(key)),
                        React.createElement('span', { key: 'value', className: 'config-value' }, formatConfigValue(value)),
                      ]
                    )
                  )
                )
              : React.createElement('p', { key: 'empty', className: 'callout-detail' }, 'No overrides captured.'),
          ]
        ),
      ]
    );
  };

  const ArchitecturePanel = ({ metrics }) => {
    const generator = metrics?.rgan?.architecture?.generator || [];
    const discriminator = metrics?.rgan?.architecture?.discriminator || [];
    const lstm = metrics?.lstm?.architecture || [];

    if (!generator.length && !discriminator.length && !lstm.length) return null;

    const renderList = (title, items) =>
      React.createElement(
        'article',
        { key: title, className: 'card architecture-card' },
        [
          React.createElement('h3', { key: 'title' }, title),
          React.createElement(
            'ol',
            { key: 'list' },
            items.map((item, index) => React.createElement('li', { key: `${title}-${index}` }, item))
          ),
        ]
      );

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Model Architectures'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Layer-by-layer descriptions captured at experiment time.'
          ),
        ]),
        React.createElement(
          'div',
          { key: 'content', className: 'architecture-grid' },
          [
            generator.length ? renderList('Generator', generator) : null,
            discriminator.length ? renderList('Discriminator', discriminator) : null,
            lstm.length ? renderList('LSTM Baseline', lstm) : null,
          ].filter(Boolean)
        ),
      ]
    );
  };

  const ArtifactLinks = ({ metrics }) => {
    const entries = [];

    const pushArtifact = (title, path, description) => {
      if (!path) return;
      const absolute = path.startsWith('http') ? path : path;
      entries.push({ title, path: absolute, description });
    };

    pushArtifact('Metrics JSON', DEFAULT_METRICS_PATH, 'Raw metrics produced by the experiment runner.');

    const addPair = (label, staticPath, interactivePath) => {
      pushArtifact(`${label} (PNG)`, staticPath, 'High-resolution static figure.');
      pushArtifact(`${label} (Interactive)`, interactivePath, 'Interactive Plotly rendition.');
    };

    addPair('R-GAN curve', metrics?.rgan?.curve, metrics?.rgan?.curve_interactive);
    addPair('LSTM curve', metrics?.lstm?.curve, metrics?.lstm?.curve_interactive);
    addPair('Naïve baseline curve', metrics?.naive_baseline?.curve, metrics?.naive_baseline?.curve_interactive);
    addPair('Naïve Bayes curve', metrics?.naive_bayes?.curve, metrics?.naive_bayes?.curve_interactive);
    addPair('Model comparison (test)', metrics?.compare_plots?.test, metrics?.compare_plots?.test_interactive);
    addPair('Model comparison (train)', metrics?.compare_plots?.train, metrics?.compare_plots?.train_interactive);
    addPair('Baseline comparison (naïve vs bayes)', metrics?.compare_plots?.naive_comparison, metrics?.compare_plots?.naive_comparison_interactive);
    addPair('Classical baselines', metrics?.classical?.curves, metrics?.classical?.curves_interactive);
    addPair('Learning curves', metrics?.learning_curves?.plot, metrics?.learning_curves?.plot_interactive);

    if (!entries.length) return null;

    return React.createElement(
      'section',
      { className: 'section' },
      [
        React.createElement('header', { key: 'header', className: 'section-header' }, [
          React.createElement('h2', { key: 'title' }, 'Artifacts'),
          React.createElement(
            'p',
            { key: 'subtitle', className: 'section-subtitle' },
            'Download the raw metrics and high-resolution figures referenced throughout the dashboard.'
          ),
        ]),
        React.createElement(
          'ul',
          { key: 'list', className: 'artifact-list' },
          entries.map((entry) =>
            React.createElement(
              'li',
              { key: `${entry.title}-${entry.path}` },
              [
                React.createElement('a', { key: 'link', href: entry.path, target: '_blank', rel: 'noopener noreferrer' }, entry.title),
                React.createElement('span', { key: 'desc' }, entry.description),
              ]
            )
          )
        ),
      ]
    );
  };

  const Footer = () =>
    React.createElement(
      'footer',
      { className: 'page-footer' },
      React.createElement('span', null, 'RGAN Research Dashboard · Built with React & Plotly')
    );

  const App = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
      let cancelled = false;
      setLoading(true);
      setError(null);

      fetch(DEFAULT_METRICS_PATH, { cache: 'no-store' })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to load metrics (${response.status} ${response.statusText})`);
          }
          return response.json();
        })
        .then((data) => {
          if (!cancelled) {
            setMetrics(data);
            setLoading(false);
          }
        })
        .catch((err) => {
          if (!cancelled) {
            console.error('Dashboard metrics load failure', err);
            setError(err.message || 'Unexpected error while loading metrics');
            setLoading(false);
          }
        });

      return () => {
        cancelled = true;
      };
    }, []);

    return React.createElement(
      'div',
      { className: 'dashboard' },
      [
        React.createElement(
          'header',
          { key: 'hero', className: 'hero' },
          [
            React.createElement('div', { key: 'text', className: 'hero-text' }, [
              React.createElement('p', { key: 'eyebrow', className: 'hero-eyebrow' }, 'R-GAN Research Project'),
              React.createElement('h1', { key: 'title' }, 'Interactive Experiment Dashboard'),
              React.createElement(
                'p',
                { key: 'lead', className: 'hero-lead' },
                'An executive view of training dynamics, baselines, and robustness analyses for reproducible R-GAN experiments.'
              ),
            ]),
            React.createElement(
              'div',
              { key: 'status', className: 'hero-status' },
              loading
                ? React.createElement('span', { className: 'status-pill status-loading' }, 'Loading metrics...')
                : error
                ? React.createElement('span', { className: 'status-pill status-error' }, `Error: ${error}`)
                : React.createElement('span', { className: 'status-pill status-ready' }, 'Metrics loaded'),
              React.createElement(
                'code',
                { key: 'path', className: 'metrics-path' },
                `Source: ${DEFAULT_METRICS_PATH}`
              )
            ),
          ]
        ),
        loading
          ? React.createElement('section', { key: 'loading', className: 'section loading-section' }, [
              React.createElement('div', { key: 'spinner', className: 'spinner' }),
              React.createElement('p', { key: 'label' }, 'Crunching the latest metrics...'),
            ])
          : null,
        !loading && !error && metrics
          ? React.createElement(
              React.Fragment,
              { key: 'content' },
              [
                React.createElement(DatasetOverview, { key: 'overview', metrics }),
                    React.createElement(HighlightsGrid, { key: 'highlights', metrics }),
                React.createElement(SummaryCards, { key: 'summary', metrics }),
                    React.createElement(TrainingHistorySection, { key: 'history', metrics }),
                React.createElement(ComparisonBarChart, { key: 'comparison', metrics }),
                React.createElement(LearningCurveChart, { key: 'learning', metrics }),
                React.createElement(NoiseRobustnessChart, { key: 'noise', metrics }),
                React.createElement(PrecisionTable, { key: 'precision', metrics }),
                    React.createElement(ConfigurationPanel, { key: 'config', metrics }),
                    React.createElement(TuningSummary, { key: 'tuning', metrics }),
                React.createElement(ArchitecturePanel, { key: 'architecture', metrics }),
                React.createElement(ArtifactLinks, { key: 'artifacts', metrics }),
              ]
            )
          : null,
        React.createElement(Footer, { key: 'footer' }),
      ]
    );
  };

  ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
})();
