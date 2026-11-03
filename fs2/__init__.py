import os

# Colab sets MPLBACKEND to a notebook-only backend that fails in plain scripts
os.environ["MPLBACKEND"] = "Agg"
