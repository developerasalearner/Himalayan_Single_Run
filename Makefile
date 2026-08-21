# =====================================================================
# Convenience targets. Run `make help` for the list.
# =====================================================================

PYTHON ?= python3
SYNTHETIC = --set paths.expeditions_dir=data/synthetic/himalayan-expeditions \
            --set paths.weather_dir=data/synthetic/weather

.PHONY: help install data all dataset baselines selection imbalance final ablation \
        test smoke clean clean-results clean-all

help:
	@echo "Setup"
	@echo "  make install       install dependencies from requirements.txt"
	@echo "  make data          download both archives (needs Kaggle credentials)"
	@echo ""
	@echo "Run"
	@echo "  make all           every stage, in order"
	@echo "  make dataset       stage 1 - build the modelling table"
	@echo "  make baselines     stage 2 - Table I"
	@echo "  make selection     stage 3 - Table II, freezes feature subsets"
	@echo "  make imbalance     stage 4 - Table III"
	@echo "  make final         stage 5 - Table IV, final metrics, figures"
	@echo "  make ablation      stage 6 - leakage ablation"
	@echo ""
	@echo "Verify"
	@echo "  make test          fast unit tests, no data needed"
	@echo "  make smoke         full pipeline on synthetic data"
	@echo ""
	@echo "Clean"
	@echo "  make clean-results remove results and figures"
	@echo "  make clean         also remove the processed-data cache"
	@echo "  make clean-all     also remove frozen feature subsets"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/00_download_data.py

all:
	$(PYTHON) scripts/run_all.py

dataset:
	$(PYTHON) scripts/01_build_dataset.py

baselines:
	$(PYTHON) scripts/02_baselines.py

selection:
	$(PYTHON) scripts/03_feature_selection.py

imbalance:
	$(PYTHON) scripts/04_imbalance.py

final:
	$(PYTHON) scripts/05_calibration_final.py

ablation:
	$(PYTHON) scripts/06_leakage_ablation.py

test:
	$(PYTHON) -m pytest tests/ -q

# Exercises every stage against a synthetic stand-in. Proves the code
# path works; proves nothing about the science.
smoke:
	$(PYTHON) scripts/make_synthetic_data.py --out data/synthetic
	$(PYTHON) scripts/run_all.py --quiet --research $(SYNTHETIC) \
		--set selection.population_size=6 \
		--set selection.iterations=3 \
		--set selection.embedded.cv_folds=3
	@echo ""
	@echo "Smoke test complete. Synthetic data: results are meaningless by design."

clean-results:
	rm -rf results/*.csv results/*.json figures/*.png

clean: clean-results
	rm -rf data/processed data/synthetic
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -f artifacts/selected_features.json
