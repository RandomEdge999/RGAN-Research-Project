# RGAN Analytics Terminal

A professional React dashboard for visualizing R-GAN experiment results. Features a premium "Trading Terminal" aesthetic designed for financial time-series analysis.

## Stack

- **React 18** with Vite for fast development
- **Recharts** for interactive charts
- **Lucide Icons** for modern iconography
- **react-dropzone** for file upload handling
- **Vanilla CSS** with CSS variables for theming

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Usage

1. Run an experiment using `run_experiment.py` to generate a `metrics.json` file
2. Start the dashboard with `npm run dev`
3. Open `http://localhost:5173` in your browser
4. Drag and drop your `metrics.json` file into the upload area

## Features

- **Performance Summary** – Key KPIs including best RMSE and baseline comparison
- **Detailed Metrics Table** – RMSE, MAE, sMAPE, MASE for all models
- **Original Scale Metrics** – Unscaled error values for real-world interpretation
- **Training Dynamics** – Interactive D/G loss and validation RMSE curves
- **Noise Robustness Analysis** – Model performance across noise perturbation levels
- **Statistical Confidence** – 95% bootstrap confidence intervals

## File Structure

```
src/
├── App.jsx          # Root component
├── Dashboard.jsx    # Main dashboard component
├── index.css        # Trading terminal theme styles
└── main.jsx         # Entry point
```
