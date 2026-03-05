CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_symbol_timeframe
ON ohlcv(symbol, timeframe);
