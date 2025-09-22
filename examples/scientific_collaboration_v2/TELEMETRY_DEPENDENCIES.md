# OpenTelemetry Dependencies for Scientific Collaboration V2

This enhanced version of the scientific collaboration simulation includes OpenTelemetry tracing support. To use the telemetry features, install the following dependencies:

## Required OpenTelemetry Packages

```bash
pip install opentelemetry-sdk>=1.34.1
pip install opentelemetry-exporter-otlp-proto-grpc  # For GRPC export
# OR
pip install opentelemetry-exporter-otlp-proto-http  # For HTTP export
```

## Current Dependencies in autogen-core

The `autogen-core` package already includes:
- `opentelemetry-api>=1.34.1` (required dependency)
- `opentelemetry-sdk>=1.34.1` (dev dependency)

## Fallback Behavior

If OpenTelemetry SDK is not available, the simulation will:
1. Display a warning message
2. Continue running without telemetry tracing
3. Still provide enhanced logging functionality

## Features Enabled with Telemetry

When OpenTelemetry is available, the simulation provides:
- Agent creation and invocation spans
- Message handling tracing
- Collaboration event tracking
- Console span export for demo purposes

## Example Usage

```bash
# Install dependencies
pip install opentelemetry-sdk

# Run the enhanced simulation
python main.py --verbose
```

The verbose flag enables detailed trace and event logging to both console and file.