# Marketing Mix Modelling (MMM) Architecture

## Overview

End-to-end MMM pipeline with multiple model types, configurable transforms, and constraints.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MMM FRONTEND (Streamlit)                         │
│  • Dataset upload / sample data                                           │
│  • Transforms: decay, curvature, half-saturation                          │
│  • Model selection: Linear, Ridge, Lasso, Bayesian, Hierarchical          │
│  • Constraints: positive coefficients, lag sum range                      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           MMM PIPELINE                                    │
│  • Data validation                                                        │
│  • Transform application (adstock → saturation)                           │
│  • Model fitting with constraints                                         │
│  • Prediction & attribution                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ TRANSFORMS      │      │ MODELS              │      │ CONSTRAINTS         │
│ • Adstock       │      │ • Linear            │      │ • Positive coeffs   │
│ • Saturation   │      │ • Ridge / Lasso     │      │ • Lag sum [lb, ub]   │
│   (Hill curve)  │      │ • Bayesian          │      │                     │
│                 │      │ • Hierarchical      │      │                     │
└─────────────────┘      └─────────────────────┘      └─────────────────────┘
```

## Components

| Module | Purpose |
|--------|---------|
| `transforms.py` | Adstock (carryover) and Hill saturation |
| `config.py` | MMMConfig dataclass |
| `models/linear.py` | Linear regression with scipy.optimize |
| `models/ridge_lasso.py` | Ridge/Lasso with constraints |
| `models/bayesian.py` | PyMC Bayesian model |
| `models/hierarchical.py` | PyMC hierarchical (partial pooling) |
| `pipeline.py` | Orchestrates transforms + model |

## Run

```bash
# Generate sample data
python create_mmm_dataset.py

# Start frontend
streamlit run mmm_app.py

# Or use the script
./run_mmm_app.sh
```

## Constraints

- **Positive constraints**: All channel coefficients ≥ 0
- **Lag sum**: Sum of channel coefficients in [lower, upper] range
