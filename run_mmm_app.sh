#!/bin/bash
# Run Marketing Mix Modelling frontend
cd "$(dirname "$0")"
streamlit run mmm_app.py --server.port 8502
