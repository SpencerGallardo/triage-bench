.PHONY: install test eval eval-judge eval-fast report tool-use clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

# Full eval, all 40 cases × 4 strategies = 160 calls
eval:
	python src/run_eval.py --out results/latest
	python src/generate_report.py --results-dir results/latest

# Full eval + judge (~320 calls total, slower and costlier)
eval-judge:
	python src/run_eval.py --judge --out results/with_judge
	python src/generate_report.py --results-dir results/with_judge

# Smoke test — 5 cases × 4 strategies
eval-fast:
	python src/run_eval.py --limit 5 --out results/smoke
	python src/generate_report.py --results-dir results/smoke

# Stretch goal: tool-use variant
tool-use:
	python src/run_tool_use_eval.py --out results/tool_use

clean:
	rm -rf results/smoke results/latest results/with_judge results/tool_use
