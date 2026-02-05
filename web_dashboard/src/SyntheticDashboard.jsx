/**
 * SyntheticDashboard Component
 *
 * Displays synthetic data analysis:
 * 1. Classification Metrics Table (Table 1) - Real vs Synthetic Detection
 * 2. Synthetic Quality Table (Table 2) - Fréchet Distance, Variance Diff, Discrimination Score
 * 3. Data Augmentation Table (Table 3) - Model Performance Comparison
 * 4. Real vs Synthetic Sequences Chart - Time series overlay visualization
 *
 * This is a standalone component that can be optionally integrated into Dashboard.jsx
 * or used as a separate view.
 */

import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Table, Zap, TrendingUp, LineChart as LineChartIcon } from 'lucide-react';

const fmt = (val) => {
  if (val === null || val === undefined) return '-';
  if (typeof val === 'number') return val.toFixed(4);
  return val;
};

const SyntheticDashboard = ({ metrics }) => {
  // Prepare chart data for real vs synthetic sequences
  const comparisonChartData = useMemo(() => {
    if (!metrics?.comparison_data) return [];

    const { real_sequences = [], synthetic_sequences = [] } = metrics.comparison_data;

    if (!real_sequences.length || !synthetic_sequences.length) return [];

    // Use first sequence
    const real = real_sequences[0];
    const synthetic = synthetic_sequences[0];

    if (!Array.isArray(real) || !Array.isArray(synthetic)) return [];

    return real.slice(0, 50).map((val, idx) => ({
      timestep: idx,
      real: Number(val) || 0,
      synthetic: Number(synthetic[idx]) || 0,
    }));
  }, [metrics]);

  // ========================================================================
  // TABLE 1: Classification Metrics (Real vs Synthetic Detection)
  // ========================================================================
  const renderClassificationMetricsTable = () => {
    if (!metrics?.classification_metrics) return null;

    const metrics_table = metrics.classification_metrics;

    if (Object.keys(metrics_table).length === 0) return null;

    return (
      <div className="panel">
        <div className="panel-header">
          <span><Table size={14} /> Classification Metrics (Real vs Synthetic Detection)</span>
        </div>
        <div className="panel-body !p-0">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="text-right">Accuracy</th>
                  <th className="text-right">F1</th>
                  <th className="text-right">Precision</th>
                  <th className="text-right">Recall</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics_table).map(([model, scores]) => (
                  <tr key={model}>
                    <td className="font-mono text-sm">{model}</td>
                    <td className="text-right font-mono text-sm">{fmt(scores.accuracy)}</td>
                    <td className="text-right font-mono text-sm">{fmt(scores.f1)}</td>
                    <td className="text-right font-mono text-sm">{fmt(scores.precision)}</td>
                    <td className="text-right font-mono text-sm">{fmt(scores.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  // ========================================================================
  // TABLE 2: Synthetic Quality Metrics
  // ========================================================================
  const renderSyntheticQualityTable = () => {
    if (!metrics?.synthetic_quality) return null;

    const quality = metrics.synthetic_quality;

    if (Object.keys(quality).length === 0) return null;

    return (
      <div className="panel">
        <div className="panel-header">
          <span><Zap size={14} /> Synthetic Data Quality Metrics</span>
        </div>
        <div className="panel-body !p-0">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>GAN Type</th>
                  <th className="text-right">Fréchet Distance</th>
                  <th className="text-right">Variance Diff Abs</th>
                  <th className="text-right">Variance Diff %</th>
                  <th className="text-right">Discrimination Acc</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(quality).map(([gan_type, metrics_data]) => (
                  <tr key={gan_type}>
                    <td className="font-mono text-sm">{gan_type}</td>
                    <td className="text-right font-mono text-sm">
                      {fmt(metrics_data.frechet_distance)}
                    </td>
                    <td className="text-right font-mono text-sm">
                      {fmt(metrics_data.variance_difference?.abs_diff)}
                    </td>
                    <td className="text-right font-mono text-sm">
                      {metrics_data.variance_difference?.rel_diff?.toFixed(2)}%
                    </td>
                    <td className="text-right font-mono text-sm">
                      {fmt(metrics_data.discrimination_score?.accuracy)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  // ========================================================================
  // TABLE 3: Data Augmentation Effectiveness
  // ========================================================================
  const renderDataAugmentationTable = () => {
    if (!metrics?.data_augmentation) return null;

    const augmentation = metrics.data_augmentation;

    if (Object.keys(augmentation).length === 0) return null;

    return (
      <div className="panel">
        <div className="panel-header">
          <span><TrendingUp size={14} /> Data Augmentation Effectiveness</span>
        </div>
        <div className="panel-body !p-0">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="text-right">Real Only RMSE</th>
                  <th className="text-right">Real+Synthetic RMSE</th>
                  <th className="text-right">Improvement %</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(augmentation).map(([model, results]) => {
                  const real_rmse = results.real_only?.rmse || 0;
                  const mixed_rmse = results.real_plus_synthetic?.rmse || 0;
                  const improvement = real_rmse > 0
                    ? ((real_rmse - mixed_rmse) / real_rmse * 100)
                    : 0;

                  return (
                    <tr key={model}>
                      <td className="font-mono text-sm">{model}</td>
                      <td className="text-right font-mono text-sm">{fmt(real_rmse)}</td>
                      <td className="text-right font-mono text-sm">{fmt(mixed_rmse)}</td>
                      <td
                        className="text-right font-mono text-sm"
                        style={{
                          color: improvement > 0 ? '#10b981' : '#ef4444',
                          fontWeight: 'bold'
                        }}
                      >
                        {improvement.toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  // ========================================================================
  // CHART: Real vs Synthetic Sequences
  // ========================================================================
  const renderSequenceChart = () => {
    if (!comparisonChartData || comparisonChartData.length === 0) return null;

    return (
      <div className="panel h-[350px]">
        <div className="panel-header">
          <span><LineChartIcon size={14} /> Real vs Synthetic Sequences (First Sample)</span>
        </div>
        <div className="panel-body">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={comparisonChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timestep" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#1e293b',
                  color: '#e2e8f0'
                }}
                formatter={(val) => val.toFixed(4)}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="real"
                stroke="#06b6d4"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="Real"
              />
              <Line
                type="monotone"
                dataKey="synthetic"
                stroke="#ec4899"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                name="Synthetic"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // ========================================================================
  // RENDER MAIN COMPONENT
  // ========================================================================
  const hasContent =
    metrics?.classification_metrics ||
    metrics?.synthetic_quality ||
    metrics?.data_augmentation ||
    comparisonChartData.length > 0;

  if (!hasContent) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>Synthetic Data Analysis</span>
        </div>
        <div className="panel-body" style={{ textAlign: 'center', color: '#64748b' }}>
          <p>No synthetic analysis data available. Run augmentation experiment to generate results.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="synthetic-dashboard">
      {renderClassificationMetricsTable()}
      {renderSyntheticQualityTable()}
      {renderDataAugmentationTable()}
      {renderSequenceChart()}
    </div>
  );
};

export default SyntheticDashboard;
