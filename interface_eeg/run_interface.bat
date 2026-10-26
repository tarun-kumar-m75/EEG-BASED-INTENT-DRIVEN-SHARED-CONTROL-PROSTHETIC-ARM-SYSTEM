@echo off
title EEG Prosthetic Arm Interface
echo ========================================================
echo Launching EEG Intent-Driven Prosthetic Arm Interface...
echo Opening in default browser at http://localhost:8501
echo ========================================================
python -m streamlit run dashboard/app.py
pause
