"""
Run the full pipeline end to end:

    python run_pipeline.py

1. src.build_dataset  – ingest, validate, clean, engineer features, join, write data/processed + data quality report
2. src.analytics      – portfolio profiling, cross-cuts, insights, figures
3. src.modeling       – train/evaluate logistic regression and LightGBM, SHAP, score all loans, model report

Then start the dashboard with:  streamlit run app/dashboard.py
"""
import time

from src import analytics, build_dataset, modeling

if __name__ == "__main__":
    for name, fn in [("build_dataset", lambda: build_dataset.build()), ("analytics", analytics.run), ("modeling", modeling.run)]:
        t0 = time.time()
        print(f"\n=== {name} ===")
        fn()
        print(f"--- {name} finished in {time.time() - t0:.0f}s")
    print("\nPipeline complete. Launch the dashboard with: streamlit run app/dashboard.py")
