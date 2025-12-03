import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area
} from 'recharts';
import { Activity, Database, Server, Terminal, TrendingUp } from 'lucide-react';

const fmt = (num) => {
  if (num === undefined || num === null) return '-';
  return typeof num === 'number' ? num.toFixed(8) : num;
};

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/data/metrics.json')
      .then(res => res.json())
      .then(jsonData => {
        setData(jsonData);
        setLoading(false);
      })
      .catch(err => console.error("Failed to load metrics:", err));
  }, []);

  if (loading) return <div className="dashboard-container items-center justify-center text-secondary font-mono">INITIALIZING SYSTEM...</div>;
  if (!data) return <div className="dashboard-container items-center justify-center text-red font-mono">DATA LOAD ERROR</div>;

  // Prepare Data
  const history = data.rgan.history.epoch.map((e, i) => ({
    epoch: e,
    d_loss: data.rgan.history.D_loss[i],
    g_loss: data.rgan.history.G_loss[i],
    train_rmse: data.rgan.history.train_rmse[i],
    test_rmse: data.rgan.history.test_rmse[i],
    val_rmse: data.rgan.history.val_rmse[i],
  }));

  const models = [
    { id: 'rgan', name: 'RGAN (Proposed)', color: '#06b6d4', data: data.rgan.test },
    { id: 'lstm', name: 'LSTM (Supervised)', color: '#3b82f6', data: data.lstm.test },
    { id: 'arima', name: 'ARIMA', color: '#f97316', data: data.noise_robustness[0].arima },
    { id: 'arma', name: 'ARMA', color: '#ef4444', data: data.noise_robustness[0].arma },
    { id: 'naive', name: 'Naïve Baseline', color: '#64748b', data: data.naive_baseline.test },
  ];

  const robustnessData = data.noise_robustness.map(item => ({
    noise: item.sd,
    rgan: item.rgan.rmse,
    lstm: item.lstm.rmse,
    arima: item.arima.rmse,
    arma: item.arma.rmse,
  }));

  return (
    <div className="dashboard-container">
      {/* Top Bar */}
      <header className="top-bar">
        <div className="app-title">
          <Terminal size={20} className="text-blue" />
          <span>RGAN ANALYTICS TERMINAL <span className="text-muted text-sm font-normal">v2.0.4</span></span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-xs text-muted">DATASET</div>
            <div className="font-mono text-xs">{data.dataset}</div>
          </div>
          <div className="status-indicator">SYSTEM ONLINE</div>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">

        {/* Left Column: Metrics & Config */}
        <div className="flex flex-col gap-4">

          {/* Key Metrics Panel */}
          <div className="panel">
            <div className="panel-header">
              <span><Activity size={14} className="inline mr-2" /> Performance Summary</span>
            </div>
            <div className="panel-body">
              <div className="metric-grid">
                <div className="metric-card">
                  <div className="metric-label">Best RMSE (Test)</div>
                  <div className="metric-value text-blue">{fmt(data.rgan.test.rmse)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Baseline RMSE</div>
                  <div className="metric-value text-muted">{fmt(data.naive_baseline.test.rmse)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Training Time</div>
                  <div className="metric-value">00:02:14</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Epochs</div>
                  <div className="metric-value">{history.length}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Model Table */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span><Database size={14} className="inline mr-2" /> Detailed Metrics</span>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="text-right">RMSE</th>
                    <th className="text-right">MAE</th>
                    <th className="text-right">sMAPE</th>
                    <th className="text-right">MASE</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map(m => (
                    <tr key={m.id}>
                      <td className="font-bold" style={{ color: m.color }}>{m.name}</td>
                      <td className="text-right val-mono">{fmt(m.data.rmse)}</td>
                      <td className="text-right val-mono">{fmt(m.data.mae)}</td>
                      <td className="text-right val-mono">{fmt(m.data.smape)}</td>
                      <td className="text-right val-mono">{fmt(m.data.mase)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Original Scale Metrics */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span><Server size={14} className="inline mr-2" /> Original Scale (Unscaled)</span>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="text-right">RMSE (Orig)</th>
                    <th className="text-right">MAE (Orig)</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map(m => (
                    <tr key={m.id}>
                      <td className="font-bold" style={{ color: m.color }}>{m.name}</td>
                      <td className="text-right val-mono">{fmt(m.data.rmse_orig)}</td>
                      <td className="text-right val-mono">{fmt(m.data.mae_orig)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>

        {/* Right Column: Charts */}
        <div className="flex flex-col gap-4">

          {/* Training Dynamics */}
          <div className="panel" style={{ height: '300px' }}>
            <div className="panel-header">
              <span>Training Dynamics (Loss & Error)</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2b303b" />
                  <XAxis dataKey="epoch" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#16181d', borderColor: '#2b303b', fontFamily: 'monospace' }}
                    itemStyle={{ fontSize: '12px' }}
                    formatter={(val) => val.toFixed(6)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="d_loss" stroke="#ef4444" dot={false} strokeWidth={1.5} name="D Loss" />
                  <Line type="monotone" dataKey="g_loss" stroke="#06b6d4" dot={false} strokeWidth={1.5} name="G Loss" />
                  <Line type="monotone" dataKey="val_rmse" stroke="#10b981" dot={false} strokeWidth={1.5} name="Val RMSE" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Noise Robustness */}
          <div className="panel" style={{ height: '300px' }}>
            <div className="panel-header">
              <span>Noise Robustness Analysis</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={robustnessData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2b303b" />
                  <XAxis dataKey="noise" stroke="#64748b" label={{ value: 'Noise SD', position: 'insideBottom', offset: -5, fill: '#64748b' }} />
                  <YAxis stroke="#64748b" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#16181d', borderColor: '#2b303b', fontFamily: 'monospace' }}
                    formatter={(val) => val.toFixed(6)}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="rgan" stroke="#06b6d4" strokeWidth={2} name="RGAN" />
                  <Line type="monotone" dataKey="lstm" stroke="#3b82f6" strokeWidth={2} name="LSTM" />
                  <Line type="monotone" dataKey="arima" stroke="#f97316" strokeWidth={2} strokeDasharray="3 3" name="ARIMA" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confidence Intervals Table */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span>Statistical Confidence (95% CI)</span>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th className="text-right">RMSE Low</th>
                    <th className="text-right">RMSE High</th>
                    <th className="text-right">MAE Low</th>
                    <th className="text-right">MAE High</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map(m => (
                    <tr key={m.id}>
                      <td className="font-bold" style={{ color: m.color }}>{m.name}</td>
                      <td className="text-right val-mono">{fmt(m.data.rmse_ci_low)}</td>
                      <td className="text-right val-mono">{fmt(m.data.rmse_ci_high)}</td>
                      <td className="text-right val-mono">{fmt(m.data.mae_ci_low)}</td>
                      <td className="text-right val-mono">{fmt(m.data.mae_ci_high)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Dashboard;
