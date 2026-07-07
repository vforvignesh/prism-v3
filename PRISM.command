#!/bin/zsh
# Double-click to launch the PRISM dashboard (opens in your browser).
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run app.py
