##############################################################################
# Python Playwright Tests (chatbot_tests)
##############################################################################

.PHONY: install
install:		## Install Python dependencies for chatbot tests
	pip install -e .
	playwright install chromium

.PHONY: run
run:		## Run all chatbot Playwright tests
	python -m chatbot_tests.main run

.PHONY: headed
headed:		## Run tests with visible browser
	python -m chatbot_tests.main run --headed

.PHONY: server
server:		## Start the chatbot tests API server
	python -m chatbot_tests.main serve --port 8000

.PHONY: report
report:		## View latest test report
	python -m chatbot_tests.main report --latest --open