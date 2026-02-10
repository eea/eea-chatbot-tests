##############################################################################
# EEA Chatbot Tests
##############################################################################

.PHONY: install
install:		## Install dependencies and Playwright browser
	pip install -e .
	playwright install chromium

.PHONY: run
run:			## Run all chatbot tests
	chatbot_tests run

.PHONY: headed
headed:			## Run tests with visible browser
	chatbot_tests run --headed

.PHONY: analyze
analyze:		## Analyze latest test report (usage: make analyze FILE=./reports/test_run_*.jsonl)
	chatbot_tests analyze $(FILE) --all

.PHONY: compare
compare:		## Compare test runs (usage: make compare FILES="./run1.jsonl ./run2.jsonl")
	chatbot_tests compare $(FILES)

.PHONY: help
help:			## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
