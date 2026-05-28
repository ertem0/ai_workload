PYTHON ?= .venv/bin/python

WORKLOAD ?=
WORKLOADS ?= $(WORKLOAD)
EXPERT_TRACE ?=
JSON_OUTPUT ?=
OUTPUT_DIR ?=
COMPACT ?= 0

COMPACT_FLAG := $(if $(filter 1 true yes,$(COMPACT)),--compact,)

.PHONY: help workload_metrics expert_metrics trace_to_json

help:
	@echo "Available targets:"
	@echo "  make workload_metrics WORKLOAD=results/.../workload_trace.pkl [OUTPUT_DIR=metrics]"
	@echo "  make workload_metrics WORKLOADS=\"trace1.pkl trace2.pkl\" [OUTPUT_DIR=metrics]"
	@echo "  make expert_metrics EXPERT_TRACE=results/.../expert_traces_raw.pkl [OUTPUT_DIR=results/...]"
	@echo "  make trace_to_json WORKLOAD=results/.../workload_trace.pkl [JSON_OUTPUT=trace.json] [COMPACT=1]"

workload_metrics:
	@if [ -z "$(strip $(WORKLOADS))" ]; then \
		echo "Usage: make workload_metrics WORKLOAD=results/.../workload_trace.pkl"; \
		echo "   or: make workload_metrics WORKLOADS=\"trace1.pkl trace2.pkl\""; \
		exit 2; \
	fi
	$(PYTHON) -m src.metrics.workload_metrics $(WORKLOADS) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

expert_metrics:
	@if [ -z "$(strip $(EXPERT_TRACE))" ]; then \
		echo "Usage: make expert_metrics EXPERT_TRACE=results/.../expert_traces_raw.pkl"; \
		exit 2; \
	fi
	$(PYTHON) run_experts_report.py $(EXPERT_TRACE) $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),)

trace_to_json:
	@if [ -z "$(strip $(WORKLOAD))" ]; then \
		echo "Usage: make trace_to_json WORKLOAD=results/.../workload_trace.pkl"; \
		exit 2; \
	fi
	$(PYTHON) scripts/workload_trace_to_json.py $(WORKLOAD) $(if $(JSON_OUTPUT),--output $(JSON_OUTPUT),) $(COMPACT_FLAG)
