# MMM - Marketing Mix Modelling

Marketing Mix Modelling app with multiple model types, configurable transforms, and constraints.

## Features

- **Models:** Linear, Ridge, Lasso, Bayesian, Hierarchical
- **Transforms:** Adstock (carryover), Saturation (Hill curve)
- **Constraints:** Positive coefficients, lag sum range
- **Dynamic column mapping:** Works with any dataset

## Run locally

```bash
pip install -r requirements.txt
streamlit run mmm_app.py
```

## Deploy on Streamlit Cloud

1. Connect this repo at [share.streamlit.io](https://share.streamlit.io)
2. Set **Main file path** to `mmm_app.py`
3. Deploy

## Sample data

Use the built-in sample dataset or upload your own CSV. The app infers columns automatically.
