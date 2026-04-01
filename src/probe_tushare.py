from __future__ import annotations

import argparse
import statistics as stats
import time
from dataclasses import dataclass
from typing import Any, Callable

import tushare as ts


@dataclass(frozen=True)
class ProbeResult:
    name: str
    rows: int
    cols: int
    elapsed_s: float
    columns: list[str]
    sample: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe real Tushare endpoint outputs, columns, and latency."
    )
    parser.add_argument("--symbol", default="600000.SH")
    parser.add_argument("--trade-date", default="20260331")
    parser.add_argument("--start-date", default="20260301")
    parser.add_argument("--end-date", default="20260331")
    parser.add_argument("--finance-start-date", default="20200101")
    parser.add_argument("--finance-end-date", default="20260331")
    parser.add_argument("--bench-runs", type=int, default=3)
    return parser


def run_probe(name: str, fn: Callable[[], Any]) -> ProbeResult:
    t0 = time.perf_counter()
    frame = fn()
    elapsed_s = time.perf_counter() - t0

    if frame is None:
        return ProbeResult(
            name=name,
            rows=0,
            cols=0,
            elapsed_s=elapsed_s,
            columns=[],
            sample=[],
        )

    columns = list(frame.columns)
    sample = frame.head(1).to_dict(orient="records")
    return ProbeResult(
        name=name,
        rows=len(frame),
        cols=len(columns),
        elapsed_s=elapsed_s,
        columns=columns,
        sample=sample,
    )


def benchmark(name: str, fn: Callable[[], Any], runs: int) -> None:
    times: list[float] = []
    rows = 0
    cols = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        frame = fn()
        times.append(time.perf_counter() - t0)
        rows = len(frame)
        cols = len(frame.columns)
    print(
        f"[bench:{name}] rows={rows} cols={cols} "
        f"mean={stats.mean(times):.3f}s min={min(times):.3f}s max={max(times):.3f}s"
    )


def main() -> None:
    args = build_parser().parse_args()
    pro = ts.pro_api()

    print(f"tushare={ts.__version__}")
    print(
        "probe_args="
        f"symbol={args.symbol}, trade_date={args.trade_date}, "
        f"start={args.start_date}, end={args.end_date}, "
        f"finance_start={args.finance_start_date}, finance_end={args.finance_end_date}"
    )

    probes: list[tuple[str, Callable[[], Any]]] = [
        (
            "stock_basic_L",
            lambda: pro.stock_basic(
                exchange="",
                list_status="L",
                fields=(
                    "ts_code,symbol,name,area,industry,market,"
                    "list_date,delist_date,list_status"
                ),
            ),
        ),
        (
            "stock_basic_D",
            lambda: pro.stock_basic(
                exchange="",
                list_status="D",
                fields=(
                    "ts_code,symbol,name,area,industry,market,"
                    "list_date,delist_date,list_status"
                ),
            ),
        ),
        (
            "stock_basic_P",
            lambda: pro.stock_basic(
                exchange="",
                list_status="P",
                fields=(
                    "ts_code,symbol,name,area,industry,market,"
                    "list_date,delist_date,list_status"
                ),
            ),
        ),
        (
            "trade_cal",
            lambda: pro.trade_cal(
                exchange="",
                start_date=args.start_date,
                end_date=args.end_date,
                fields="exchange,cal_date,is_open,pretrade_date",
            ),
        ),
        ("daily_trade_date", lambda: pro.daily(trade_date=args.trade_date)),
        (
            "daily_symbol",
            lambda: pro.daily(
                ts_code=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            ),
        ),
        (
            "daily_basic_trade_date",
            lambda: pro.daily_basic(trade_date=args.trade_date),
        ),
        (
            "daily_basic_symbol",
            lambda: pro.daily_basic(
                ts_code=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            ),
        ),
        (
            "adj_factor_trade_date",
            lambda: pro.adj_factor(trade_date=args.trade_date),
        ),
        (
            "adj_factor_symbol",
            lambda: pro.adj_factor(
                ts_code=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            ),
        ),
        (
            "stk_limit_trade_date",
            lambda: pro.stk_limit(trade_date=args.trade_date),
        ),
        (
            "stk_limit_symbol",
            lambda: pro.stk_limit(
                ts_code=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
            ),
        ),
        (
            "suspend_d_trade_date",
            lambda: pro.suspend_d(trade_date=args.trade_date),
        ),
        (
            "income_symbol",
            lambda: pro.income(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "balancesheet_symbol",
            lambda: pro.balancesheet(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "cashflow_symbol",
            lambda: pro.cashflow(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "fina_indicator_symbol",
            lambda: pro.fina_indicator(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "forecast_symbol",
            lambda: pro.forecast(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "express_symbol",
            lambda: pro.express(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        ("dividend_symbol", lambda: pro.dividend(ts_code=args.symbol)),
        (
            "fina_audit_symbol",
            lambda: pro.fina_audit(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
            ),
        ),
        (
            "fina_mainbz_symbol",
            lambda: pro.fina_mainbz(
                ts_code=args.symbol,
                start_date=args.finance_start_date,
                end_date=args.finance_end_date,
                type="P",
            ),
        ),
    ]

    for name, fn in probes:
        try:
            result = run_probe(name, fn)
            print(
                f"[{result.name}] rows={result.rows} cols={result.cols} "
                f"elapsed={result.elapsed_s:.3f}s first_cols={result.columns[:15]}"
            )
            if result.sample:
                print(result.sample)
        except Exception as exc:
            print(f"[{name}] FAIL {type(exc).__name__}: {exc}")

    print()
    print("Benchmarks")
    benchmark("daily_trade_date", lambda: pro.daily(trade_date=args.trade_date), args.bench_runs)
    benchmark(
        "daily_basic_trade_date",
        lambda: pro.daily_basic(trade_date=args.trade_date),
        args.bench_runs,
    )
    benchmark(
        "adj_factor_trade_date",
        lambda: pro.adj_factor(trade_date=args.trade_date),
        args.bench_runs,
    )
    benchmark(
        "stk_limit_trade_date",
        lambda: pro.stk_limit(trade_date=args.trade_date),
        args.bench_runs,
    )
    benchmark(
        "daily_symbol",
        lambda: pro.daily(
            ts_code=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
        ),
        args.bench_runs,
    )
    benchmark(
        "fina_indicator_symbol",
        lambda: pro.fina_indicator(
            ts_code=args.symbol,
            start_date=args.finance_start_date,
            end_date=args.finance_end_date,
        ),
        args.bench_runs,
    )


if __name__ == "__main__":
    main()
