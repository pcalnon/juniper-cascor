# juniper-cascor

[![PyPI](https://img.shields.io/pypi/v/juniper-cascor)](https://pypi.org/project/juniper-cascor/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**A Cascade-Correlation neural-network training service with live REST + WebSocket streaming.**

`juniper-cascor` is a FastAPI service that implements the Fahlman & Lebiere (1990) Cascade-Correlation
algorithm — a *constructive* network that grows itself by recruiting hidden units one at a time, each
chosen to maximise correlation with the network's remaining error. Instead of fixing the architecture
up front, CasCor builds it as training proceeds.

The service exposes a REST API for network and training control and live WebSocket streams for
real-time introspection of the growing network. It trains on datasets from
[juniper-data](https://github.com/pcalnon/juniper-data), is driven over HTTP by
[juniper-cascor-client](https://github.com/pcalnon/juniper-cascor-client), monitored live by
[juniper-canopy](https://github.com/pcalnon/juniper-canopy), and can offload candidate-unit training
to distributed [juniper-cascor-worker](https://github.com/pcalnon/juniper-cascor-worker) instances.

> **Part of the Juniper platform.** juniper-cascor is the Cascade-Correlation training backend of
> [Juniper](https://github.com/pcalnon/juniper-ml) — a multi-package ML research platform built around
> constructive and recurrent neural networks.

## Install

```bash
pip install juniper-cascor
```

For development from a clone:

```bash
git clone https://github.com/pcalnon/juniper-cascor.git && cd juniper-cascor
pip install -e ".[ml,api,observability]"
```

## Run

```bash
python src/server.py                          # binds 127.0.0.1:8200
curl http://localhost:8200/v1/health/ready
```

A minimal train loop over the REST API — create a network, stage a dataset, start training:

```bash
curl -sX POST localhost:8200/v1/network \
  -H 'Content-Type: application/json' -d '{"config": {"input_size": 2, "output_size": 2}}'
curl -sX POST localhost:8200/v1/training/dataset \
  -H 'Content-Type: application/json' -d '{"dataset": {"generator": "spiral"}}'
curl -sX POST localhost:8200/v1/training/start \
  -H 'Content-Type: application/json' -d '{"params": {"max_epochs": 100}}'
curl -s localhost:8200/v1/training/status
```

## API

REST routes (prefix `/v1`; responses wrapped in a `{status, data, meta}` envelope):

| Route | Methods | Purpose |
|-------|---------|---------|
| `/health`, `/health/live`, `/health/ready` | GET | Liveness / readiness probes |
| `/network` | POST / GET / DELETE | Create, inspect, delete the network |
| `/network/topology`, `/network/stats` | GET | Topology + statistics |
| `/training/start`, `/stop`, `/pause`, `/resume`, `/reset` | POST | Training control |
| `/training/status`, `/training/params` | GET | Training state + parameters |
| `/training/dataset` | POST / GET / DELETE | Stage / inspect / clear the training dataset |

WebSocket streams: `/ws/training` (live metrics + topology, resumable), `/ws/control` (control +
heartbeat), `/ws/v1/workers` (the distributed worker pool).

## Configuration

Settings load from the `JUNIPER_CASCOR_` environment namespace. Common knobs (full surface in
[`docs/install/REFERENCE.md`](docs/install/REFERENCE.md)):

| Variable | Default | Purpose |
|----------|---------|---------|
| `JUNIPER_CASCOR_HOST` / `JUNIPER_CASCOR_PORT` | `127.0.0.1` / `8200` | Bind address / port (`0.0.0.0` under Docker). |
| `JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED` | `false` | Required when binding a non-loopback interface; asserts a loopback host-publish or fronting auth layer protects the port. |
| `JUNIPER_DATA_URL` | `http://localhost:8100` | Upstream juniper-data service. |
| `JUNIPER_CASCOR_API_KEYS` | _(unset)_ | CSV `X-API-Key` values; auth disabled when unset. |
| `JUNIPER_CASCOR_LOG_LEVEL` / `_LOG_FORMAT` | `INFO` / `text` | Verbosity / `text` or `json`. |
| `JUNIPER_CASCOR_METRICS_ENABLED` | `false` | Expose `/v1/metrics` for Prometheus. |

## Docker

```bash
docker build -t juniper-cascor:latest .
docker run --rm -p 127.0.0.1:8200:8200 \
  -e JUNIPER_CASCOR_HOST=0.0.0.0 \
  -e JUNIPER_CASCOR_FRONTING_AUTH_ATTESTED=true \
  -e JUNIPER_DATA_URL=http://host.docker.internal:8100 \
  juniper-cascor:latest
```

Health is probed at `/v1/health/ready`. For the full stack, see
[`juniper-deploy`](https://github.com/pcalnon/juniper-deploy).

## Status

**Live** on PyPI. The current version is shown by the badge above; see [`CHANGELOG.md`](CHANGELOG.md).
It trains on datasets from `juniper-data` (`JUNIPER_DATA_URL`) and is driven by
`juniper-cascor-client`.

## Documentation

- [`docs/install/QUICK_START.md`](docs/install/QUICK_START.md) — installation and setup
- [`docs/install/USER_MANUAL.md`](docs/install/USER_MANUAL.md) — comprehensive usage guide
- [`docs/api/API_REFERENCE.md`](docs/api/API_REFERENCE.md) — REST + WebSocket API reference
- [`docs/install/REFERENCE.md`](docs/install/REFERENCE.md) — configuration reference

## License

MIT — see [LICENSE](./LICENSE).
