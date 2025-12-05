import React, { useEffect, useState, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area
} from 'recharts';
import { Activity, Database, Server, Terminal, TrendingUp, Upload, FileJson, AlertCircle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const fmt = (num) => {
  if (num === undefined || num === null) return '-';
  return typeof num === 'number' ? num.toFixed(8) : num;
};

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const jsonData = JSON.parse(reader.result);
        // Basic validation
        if (!jsonData.rgan || !jsonData.dataset) {
          throw new Error("Invalid metrics file format.");
        }
        setData(jsonData);
        setError(null);
      } catch (err) {
        console.error("Parse error:", err);
        setError("Failed to parse JSON file. Please ensure it is a valid RGAN metrics file.");
      }
    };
    reader.readAsText(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/json': ['.json']
    },
    maxFiles: 1
  });

  // Try to load default data if available (dev convenience), but don't block if missing
  useEffect(() => {
    fetch('/data/metrics.json')
      .then(res => {
        if (res.ok) return res.json();
        throw new Error("No default data");
      })
      .then(jsonData => {
        if (!data) setData(jsonData);
      })
      .catch(() => {
        // Silent fail - expected behavior for standalone mode
      });
  }, []);

  if (!data) {
    return (
      <div className="dashboard-container items-center justify-center min-h-screen bg-[#0d1117] text-slate-300 font-mono p-4">
        <div className="max-w-md w-full text-center space-y-8">
          <div className="space-y-2">
            <Terminal size={48} className="mx-auto text-cyan-500 mb-4" />
            <h1 className="text-2xl font-bold text-white tracking-tight">RGAN Analytics Terminal</h1>
            <p className="text-slate-400 text-sm">Upload a <code className="bg-slate-800 px-1 py-0.5 rounded text-cyan-400">metrics.json</code> file to view results.</p>
          </div>

          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-xl p-10 transition-all cursor-pointer
              ${isDragActive ? 'border-cyan-500 bg-cyan-500/10' : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/50'}
              ${error ? 'border-red-500/50 bg-red-500/5' : ''}
            `}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center gap-4">
              <div className={`p-4 rounded-full ${isDragActive ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-800 text-slate-400'}`}>
                <Upload size={24} />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-white">
                  {isDragActive ? "Drop the file here" : "Click or drag file to upload"}
                </p>
                <p className="text-xs text-slate-500">
                  Supports generated metrics.json files
                </p>
              </div>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 p-3 rounded-lg border border-red-500/20">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>
    );
  }

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
    { id: 'arima', name: 'ARIMA', color: '#f97316', data: data.noise_robustness?.[0]?.arima || {} },
    { id: 'arma', name: 'ARMA', color: '#ef4444', data: data.noise_robustness?.[0]?.arma || {} },
    { id: 'naive', name: 'Naïve Baseline', color: '#64748b', data: data.naive_baseline.test },
  ];

  const robustnessData = data.noise_robustness ? data.noise_robustness.map(item => ({
    noise: item.sd,
    rgan: item.rgan.rmse,
    lstm: item.lstm.rmse,
    arima: item.arima.rmse,
    arma: item.arma.rmse,
  })) : [];

  return (
    <div className="dashboard-container">
      {/* Top Bar */}
      <header className="top-bar">
        <div className="app-title flex items-center gap-3">
          <Terminal size={20} className="text-cyan-400" />
          <span className="font-bold tracking-tight">RGAN ANALYTICS <span className="text-slate-500 font-normal text-sm">v2.1.0</span></span>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Dataset</div>
            <div className="font-mono text-xs text-slate-300">{data.dataset}</div>
          </div>
          <button
            onClick={() => setData(null)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 transition-colors border border-slate-700"
          >
            <FileJson size={14} />
            Load New File
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">

        {/* Left Column: Metrics & Config */}
        <div className="flex flex-col gap-4">

          {/* Key Metrics Panel */}
          <div className="panel">
            <div className="panel-header">
              <span><Activity size={14} className="inline mr-2 text-cyan-400" /> Performance Summary</span>
            </div>
            <div className="panel-body">
              <div className="metric-grid">
                <div className="metric-card">
                  <div className="metric-label">Best RMSE (Test)</div>
                  <div className="metric-value text-cyan-400">{fmt(data.rgan.test.rmse)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Baseline RMSE</div>
                  <div className="metric-value text-slate-400">{fmt(data.naive_baseline.test.rmse)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Epochs</div>
                  <div className="metric-value">{history.length}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Batch Size</div>
                  <div className="metric-value">{data.rgan.config.batch_size || '-'}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Model Table */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span><Database size={14} className="inline mr-2 text-purple-400" /> Detailed Metrics</span>
            </div>
            <div className="panel-body !p-0">
              <div className="overflow-x-auto">
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
                        <td className="font-bold flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }}></div>
                          {m.name}
                        </td>
                        <td className="text-right font-mono text-slate-300">{fmt(m.data.rmse)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.mae)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.smape)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.mase)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Original Scale Metrics */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span><Server size={14} className="inline mr-2 text-green-400" /> Original Scale (Unscaled)</span>
            </div>
            <div className="panel-body !p-0">
              <div className="overflow-x-auto">
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
                        <td className="font-bold flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }}></div>
                          {m.name}
                        </td>
                        <td className="text-right font-mono text-slate-300">{fmt(m.data.rmse_orig)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.mae_orig)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Charts */}
        <div className="flex flex-col gap-4">

          {/* Training Dynamics */}
          <div className="panel h-[350px]">
            <div className="panel-header">
              <span>Training Dynamics (Loss & Error)</span>
            </div>
            <div className="panel-body">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', fontSize: '12px' }}
                    itemStyle={{ padding: 0 }}
                    formatter={(val) => val.toFixed(6)}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Line type="monotone" dataKey="d_loss" stroke="#ef4444" dot={false} strokeWidth={1.5} name="D Loss" />
                  <Line type="monotone" dataKey="g_loss" stroke="#06b6d4" dot={false} strokeWidth={1.5} name="G Loss" />
                  <Line type="monotone" dataKey="val_rmse" stroke="#10b981" dot={false} strokeWidth={1.5} name="Val RMSE" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Noise Robustness */}
          {robustnessData.length > 0 && (
            <div className="panel h-[350px]">
              <div className="panel-header">
                <span>Noise Robustness Analysis</span>
              </div>
              <div className="panel-body">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={robustnessData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="noise" stroke="#64748b" label={{ value: 'Noise SD', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 12 }} tick={{ fontSize: 12 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', fontSize: '12px' }}
                      formatter={(val) => val.toFixed(6)}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="rgan" stroke="#06b6d4" strokeWidth={2} name="RGAN" />
                    <Line type="monotone" dataKey="lstm" stroke="#3b82f6" strokeWidth={2} name="LSTM" />
                    <Line type="monotone" dataKey="arima" stroke="#f97316" strokeWidth={2} strokeDasharray="3 3" name="ARIMA" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Confidence Intervals Table */}
          <div className="panel flex-1">
            <div className="panel-header">
              <span>Statistical Confidence (95% CI)</span>
            </div>
            <div className="panel-body !p-0">
              <div className="overflow-x-auto">
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
                        <td className="font-bold flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: m.color }}></div>
                          {m.name}
                        </td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.rmse_ci_low)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.rmse_ci_high)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.mae_ci_low)}</td>
                        <td className="text-right font-mono text-slate-400">{fmt(m.data.mae_ci_high)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Dashboard;
