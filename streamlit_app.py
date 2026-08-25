"""Streamlit Community Cloud entry point.

Keeping this file at the repository root ensures the repo root is on ``sys.path`` before
loading the real application in ``app.main``. Streamlit Cloud should point its main file
path at this wrapper rather than executing ``app/main.py`` directly.
"""

from app.main import main

main()
