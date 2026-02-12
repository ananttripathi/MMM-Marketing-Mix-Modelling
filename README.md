# Marketing Mix Modelling (MMM)

A step-by-step web application for **Marketing Mix Modelling** that attributes sales/revenue to marketing channels using configurable transforms and multiple model types. All saturation transforms follow the **law of diminishing marginal returns**.

---

## 🌐 Live Application

**[🚀 Try the Live App](https://marketing-mix-modelling.streamlit.app/)**

---

## ✨ Features

### Step-by-Step Wizard
- **Step 1:** Brand name, model type, data source, column mapping (date, target, channels, segments, controls)
- **Step 2:** Per-channel transformation type, curvature, adstock parameters, model fit

### Data Sources
- **Sample dataset** – Built-in synthetic MMM data
- **Upload CSV** – Your own dataset with flexible column mapping
- **Generate dataset** – Create synthetic data with custom frequency (daily/weekly/monthly/yearly) and channel names

### Transformation Types (All Diminishing Returns)
| Transform | Formula | Default |
|-----------|---------|---------|
| **Negative exponential** | \( 1 - e^{-x/k} \) | ✓ |
| Hill | \( x^\alpha / (k^\alpha + x^\alpha) \) | |
| Log | \( \log(1 + x/k) \) | |
| Linear | \( x / (k + x) \) | |
| Power | \( x^\alpha / (k^\alpha + x^\alpha) \), α ∈ (0,1) | |

### Model Types
- **Linear** – OLS with optional constraints
- **Ridge** – L2 regularization
- **Lasso** – L1 regularization  
- **Bayesian** – PyMC with priors
- **Hierarchical** – PyMC with partial pooling

### Constraints
- **Positive coefficients** – Channel effects ≥ 0
- **Lag sum range** – Sum of channel coefficients within [lower, upper]

---

## 📁 Repository Structure

```
MMM-Marketing-Mix-Modelling/
├── mmm/                    # Core MMM package
│   ├── __init__.py
│   ├── config.py           # MMMConfig, column inference
│   ├── transforms.py      # Adstock, saturation (5 types)
│   ├── pipeline.py        # Orchestrator
│   └── models/
│       ├── base.py
│       ├── linear.py
│       ├── ridge_lasso.py
│       ├── bayesian.py     # PyMC
│       └── hierarchical.py # PyMC
├── mmm_app.py              # Streamlit step-by-step UI
├── create_mmm_dataset.py   # Synthetic data generator (CLI)
├── data/
│   └── marketing_mix_weekly.csv
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone
git clone https://github.com/ananttripathi/MMM-Marketing-Mix-Modelling.git
cd MMM-Marketing-Mix-Modelling

# Install
pip install -r requirements.txt

# Run app
streamlit run mmm_app.py
```

### Generate Sample Data (CLI)

```bash
# Weekly data, default channels
python create_mmm_dataset.py

# Monthly with custom channels
python create_mmm_dataset.py --freq monthly --channels "TV,Digital,Radio,Brand"

# Daily, negative exponential transform
python create_mmm_dataset.py --freq daily --transform negative_exponential -o my_data.csv
```

---

## 📊 Column Mapping

The app supports **dynamic column mapping** – works with any dataset. Map:

| Column | Description |
|--------|-------------|
| **Date** | Time period (week, month, etc.) |
| **Target** | Sales, revenue, or conversions |
| **Channels** | Marketing spend columns (TV, digital, etc.) |
| **Segments** | Optional – for segment-level modelling |
| **Controls** | Covariates (seasonality, holidays, promotions) |

---

## ⚙️ Deploy on Streamlit Cloud

1. Fork or use this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → Select repo, branch `main`
4. **Main file path:** `mmm_app.py`
5. Deploy

---

## 📦 Dependencies

- `streamlit` – Web UI
- `pandas`, `numpy` – Data
- `scipy` – Optimization
- `plotly` – Charts
- `pymc`, `arviz` – Bayesian/Hierarchical models (optional – remove for faster deploy if only using Linear/Ridge/Lasso)

---

## 📄 License

MIT
