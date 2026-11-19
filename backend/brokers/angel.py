"""Angel One SmartAPI broker adapter.

Login is headless: `generateSession(clientcode, pin, totp)` with a TOTP code generated from
ANGEL_TOTP_SECRET via `pyotp` - the same call the official SDK uses for algo trading, no browser
involved. Credentials come from environment variables (see `.env.example`), never hardcoded.

`get_candle_data` is a plain synchronous method (not the async `get_historical`) because the bulk
downloader (`backend/backtest_data_pipeline/angel_downloader.py`) calls it directly and synchronously,
the same way `yf_downloader.py` calls a plain `fetch_fn` - keeps the rate-limiter/retry loop simple and
testable with a fake client, independent of asyncio.

Order placement (`place_order`/`modify_order`/`cancel_order`/`positions`/`funds`) is implemented so the
*adapter* is ready, but nothing in the system calls it automatically yet - no strategy, risk manager or
scheduler is wired to trigger a live order. That's deliberate: the plan's staged gates (mass backtesting
-> paper trading -> your approval) exist precisely so a strategy proves itself before real money is at
risk, and those gates (Phase 4's risk manager, Phase 3's backtest validation) aren't built yet. Calling
these methods today only ever happens if you call them yourself, directly.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
import pyotp

from backend.brokers.base import BrokerAdapter
from backend.core.events import EventBus
from backend.core.models import OrderRequest

log = logging.getLogger(__name__)

# Angel's interval names for getCandleData, keyed by our own timeframe strings.
INTERVAL_MAP = {
    "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "1d": "ONE_DAY",
}
RAW_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# Our OrderRequest fields -> Angel's order param values.
ORDER_TYPE_MAP = {"MARKET": "MARKET", "LIMIT": "LIMIT", "SL": "STOPLOSS_LIMIT", "SL-M": "STOPLOSS_MARKET"}
PRODUCT_MAP = {"MIS": "INTRADAY", "CNC": "DELIVERY"}


class AngelLoginError(RuntimeError):
    pass


class AngelDataError(RuntimeError):
    """Raised for a well-formed-but-unsuccessful API response (status: false) that isn't a login
    problem - the caller (the downloader's retry loop) decides whether the message means "rate
    limited", "no data for this range", or something else via rate_limiter.is_rate_limit_error."""


class AngelOrderError(RuntimeError):
    """A well-formed-but-rejected order/modify/cancel response (status: false) - e.g. insufficient
    margin, invalid symbol, market closed, RMS rejection."""


class AngelOneAdapter(BrokerAdapter):
    name = "angelone"
    supports_data = True
    # True because the calls exist and work (tested against a fake client) - NOT a claim that it's safe
    # to trade with. Nothing in the system calls these automatically; see the module docstring.
    supports_trading = True

    def __init__(self, bus: EventBus, api_key: str, client_code: str, pin: str, totp_secret: str,
                client: Any | None = None, symbol_map: dict[str, dict[str, str]] | None = None):
        super().__init__(bus)
        self.api_key = api_key
        self.client_code = client_code
        self.pin = pin
        self.totp_secret = totp_secret
        self._client = client if client is not None else self._make_client(api_key)
        self.session: dict[str, Any] | None = None
        # {our_symbol: {"token": ..., "exchange": ..., "tradingsymbol": ...}} - from
        # backend.data.angel_instruments.build_symbol_map(); needed to place an order by our symbol name.
        self.symbol_map = symbol_map or {}

    @staticmethod
    def _make_client(api_key: str) -> Any:
        from SmartApi import SmartConnect  # imported lazily so tests never need the real package installed
        return SmartConnect(api_key=api_key)

    # ------------------------------------------------------------------ auth
    def login(self) -> dict[str, Any]:
        """Synchronous headless login. Raises AngelLoginError on failure."""
        totp = pyotp.TOTP(self.totp_secret).now()
        data = self._client.generateSession(self.client_code, self.pin, totp)
        if not data.get("status"):
            raise AngelLoginError(f"Angel One login failed: {data.get('message')} (code {data.get('errorcode')})")
        self.session = data["data"]
        log.info("Angel One session established", extra={"data": {"client_code": self.client_code}})
        return self.session

    # ------------------------------------------------------------------ BrokerAdapter interface
    async def connect(self) -> None:
        import asyncio
        await asyncio.to_thread(self.login)

    async def disconnect(self) -> None:
        self.session = None

    async def get_historical(self, symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
        import asyncio
        df, err = await asyncio.to_thread(
            self.get_candle_data, {"exchange": "NSE", "token": symbol}, interval,
            datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.min.time()))
        if err:
            raise AngelDataError(err)
        return df if df is not None else pd.DataFrame(columns=RAW_COLUMNS)

    # ------------------------------------------------------------------ historical data (sync, for the downloader)
    def get_candle_data(self, symbol_info: dict[str, str], interval: str, start: datetime, end: datetime
                        ) -> tuple[pd.DataFrame | None, str | None]:
        """(dataframe, None) on success, (None, error_message) on a well-formed failure response.
        Raises whatever `SmartApi.smartExceptions` exception the SDK raises for transport/auth errors -
        the downloader's retry loop catches those separately, exactly like it does for yfinance."""
        params = {
            "exchange": symbol_info["exchange"],
            "symboltoken": symbol_info["token"],
            "interval": INTERVAL_MAP[interval],
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }
        resp = self._client.getCandleData(params)
        if not resp.get("status"):
            return None, resp.get("message") or resp.get("errorcode") or "unknown error"
        rows = resp.get("data") or []
        if not rows:
            return pd.DataFrame(columns=RAW_COLUMNS), None
        df = pd.DataFrame(rows, columns=RAW_COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"], utc=False)   # Angel returns "...+05:30" - already tz-aware IST
        for c in RAW_COLUMNS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df, None

    # ------------------------------------------------------------------ trading
    def _resolve_symbol(self, symbol: str) -> dict[str, str]:
        info = self.symbol_map.get(symbol)
        if info is None:
            raise AngelOrderError(f"{symbol!r} not in symbol_map - pass symbol_map= when constructing "
                                 f"AngelOneAdapter (see backend.data.angel_instruments.build_symbol_map)")
        return info

    def _order_params(self, order: OrderRequest) -> dict[str, Any]:
        info = self._resolve_symbol(order.symbol)
        variety = "STOPLOSS" if order.order_type in ("SL", "SL-M") else "NORMAL"
        return {
            "variety": variety,
            "tradingsymbol": info["tradingsymbol"],
            "symboltoken": info["token"],
            "exchange": info["exchange"],
            "transactiontype": order.side.value,   # "BUY" | "SELL"
            "ordertype": ORDER_TYPE_MAP[order.order_type],
            "producttype": PRODUCT_MAP[order.product],
            "duration": "DAY",
            "price": order.price if order.price is not None else "0",
            "triggerprice": order.trigger_price if order.trigger_price is not None else "0",
            "quantity": str(order.qty),
            "ordertag": order.tag,
        }

    def place_order_sync(self, order: OrderRequest) -> str:
        """The real, testable, synchronous logic - places one order and returns Angel's order id.
        Raises AngelOrderError on a rejected order. `place_order` (below) is the async BrokerAdapter
        override that just wraps this in a thread; nothing calls either automatically - see the module
        docstring."""
        params = self._order_params(order)
        order_id = self._client.placeOrder(params)
        if not order_id:
            raise AngelOrderError(f"order rejected for {order.symbol}: {params}")
        log.info("order placed", extra={"data": {"symbol": order.symbol, "side": order.side.value,
                                                 "qty": order.qty, "order_id": order_id, "tag": order.tag}})
        return order_id

    def modify_order_sync(self, order_id: str, **changes: Any) -> None:
        """`changes` may include price/trigger_price/qty/order_type (same names as OrderRequest)."""
        params = {"orderid": order_id, "variety": changes.pop("variety", "NORMAL")}
        if "price" in changes:
            params["price"] = changes["price"]
        if "trigger_price" in changes:
            params["triggerprice"] = changes["trigger_price"]
        if "qty" in changes:
            params["quantity"] = str(changes["qty"])
        if "order_type" in changes:
            params["ordertype"] = ORDER_TYPE_MAP[changes["order_type"]]
        resp = self._client.modifyOrder(params)
        if not resp.get("status"):
            raise AngelOrderError(f"modify rejected for order {order_id}: {resp.get('message')}")
        log.info("order modified", extra={"data": {"order_id": order_id, "changes": changes}})

    def cancel_order_sync(self, order_id: str, variety: str = "NORMAL") -> None:
        resp = self._client.cancelOrder(order_id, variety)
        if not resp.get("status"):
            raise AngelOrderError(f"cancel rejected for order {order_id}: {resp.get('message')}")
        log.info("order cancelled", extra={"data": {"order_id": order_id}})

    def order_status_sync(self, order_id: str) -> dict[str, Any] | None:
        """This order's row from Angel One's order book: status (lower-case: complete / rejected /
        cancelled / open / ...), average fill price, filled quantity, and the broker's message."""
        resp = self._client.orderBook()
        if not resp or not resp.get("status"):
            raise AngelOrderError(f"could not fetch order book: {(resp or {}).get('message')}")
        for row in resp.get("data") or []:
            if str(row.get("orderid")) == str(order_id):
                return {"status": str(row.get("status", "")).lower(),
                        "avg_price": float(row.get("averageprice") or 0.0),
                        "filled_qty": int(float(row.get("filledshares") or 0)),
                        "message": row.get("text") or ""}
        return None

    async def wait_for_fill(self, order_id: str, timeout: float = 30.0, poll: float = 0.5) -> tuple[float, int]:
        import asyncio
        waited = 0.0
        last: dict[str, Any] | None = None
        while waited <= timeout:
            last = await asyncio.to_thread(self.order_status_sync, order_id)
            if last is not None:
                if last["status"] == "complete" and last["filled_qty"] > 0:
                    return last["avg_price"], last["filled_qty"]
                if last["status"] in ("rejected", "cancelled"):
                    raise AngelOrderError(f"order {order_id} {last['status']}: {last['message']}")
            await asyncio.sleep(poll)
            waited += poll
        raise AngelOrderError(f"order {order_id} not filled within {timeout:.0f}s (last status: {last})")

    def ltp_bulk_sync(self, exchange_tokens: dict[str, list[str]]) -> dict[str, float]:
        """Current LTP for many tokens via one REST call per exchange (chunked to Angel's 50-token cap per
        call) - used after a feed reconnect so we act on real current prices instead of stale pre-drop
        ticks while the websocket is still catching up."""
        out: dict[str, float] = {}
        for exchange, tokens in exchange_tokens.items():
            for i in range(0, len(tokens), 50):
                chunk = tokens[i:i + 50]
                resp = self._client.getMarketData("LTP", {exchange: chunk})
                if not resp or not resp.get("status"):
                    continue
                for row in (resp.get("data") or {}).get("fetched") or []:
                    out[str(row["symbolToken"])] = float(row["ltp"])
        return out

    def quotes_full_sync(self, exchange_tokens: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
        """Full market quote (LTP, OHLC, % change, traded volume, ...) per token via getMarketData("FULL"), chunked
        to Angel's 50-token cap. Used once a day by the trending-stock picker. Raises if nothing came back, so a
        rate-limit rejection is visible to the caller instead of looking like an empty market."""
        out: dict[str, dict[str, Any]] = {}
        last_msg = ""
        for exchange, tokens in exchange_tokens.items():
            for i in range(0, len(tokens), 50):
                resp = self._client.getMarketData("FULL", {exchange: tokens[i:i + 50]})
                if not resp or not resp.get("status"):
                    last_msg = str((resp or {}).get("message"))
                    continue
                for row in (resp.get("data") or {}).get("fetched") or []:
                    out[str(row["symbolToken"])] = row
        if not out:
            raise AngelOrderError(f"market quote returned nothing: {last_msg}")
        return out

    def positions_sync(self) -> list[dict[str, Any]]:
        resp = self._client.position()
        if not resp.get("status"):
            raise AngelOrderError(f"could not fetch positions: {resp.get('message')}")
        return resp.get("data") or []

    def funds_sync(self) -> dict[str, Any]:
        resp = self._client.rmsLimit()
        if not resp.get("status"):
            raise AngelOrderError(f"could not fetch funds/RMS limits: {resp.get('message')}")
        return resp.get("data") or {}

    # ------------------------------------------------------------------ BrokerAdapter async overrides
    async def place_order(self, order: OrderRequest) -> str:
        import asyncio
        return await asyncio.to_thread(self.place_order_sync, order)

    async def modify_order(self, order_id: str, **changes: Any) -> None:
        import asyncio
        await asyncio.to_thread(self.modify_order_sync, order_id, **changes)

    async def cancel_order(self, order_id: str) -> None:
        import asyncio
        await asyncio.to_thread(self.cancel_order_sync, order_id)

    async def positions(self) -> list[dict[str, Any]]:
        import asyncio
        return await asyncio.to_thread(self.positions_sync)

    async def funds(self) -> dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.funds_sync)
