.PHONY: test lint bench chart

test:
	uv run pytest -q

lint:
	uv run ruff check app tests bench

bench:
	uv run python bench/run_benchmark.py --n 300

chart:
	uv run python bench/make_chart.py

accuracy:
	uv run python -m bench.accuracy
