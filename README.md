<div align="center">

# Alpha Engine

**An event-driven, regime-adaptive, multi-alpha research and trading engine for crypto markets.**

[![tests](https://github.com/Divyansh151005/alpha-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Divyansh151005/alpha-engine/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen)
![Exchange](https://img.shields.io/badge/exchange-Binance-F0B90B?logo=binance&logoColor=white)
![Storage](https://img.shields.io/badge/storage-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Alpha Engine turns raw Binance candles into risk-managed portfolio targets through a chain of small, independent engines that talk only over an in-process event bus:

> **candles** &rarr; **features** &rarr; **market regime + normalized state** &rarr; **alpha signals** &rarr; **capital allocation** &rarr; **volatility targeting** &rarr; **drawdown control** &rarr; **final positions**

The same feature, state, and alpha code is reused by live data ingestion, a cost-aware backtester, a paper-trading loop, and a set of research and validation scripts, so research results come from the exact logic that runs live.
