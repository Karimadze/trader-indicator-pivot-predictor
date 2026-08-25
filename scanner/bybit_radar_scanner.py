# -*- coding: utf-8 -*-
"""
Reversal Radar Scanner для Bybit — TradFi перпы и xStocks, таймфреймы 1ч и 4ч.

Что делает:
  * сам получает список инструментов с Bybit:
      linear + symbolType == "stock"    -> TradFi перпетуалы (AAPLUSDT, MRVLUSDT, ...)
      spot   + symbolType == "xstocks"  -> токенизированные акции (AAPLXUSDT, ...)
  * качает свечи 1ч и 4ч, считает те же 7 компонентов, что и ReversalRadar.pine;
  * сообщает о сигнале СРАЗУ, ещё на незакрытой свече (пометка FORMING),
    и отдельно — когда свеча закрылась (пометка CLOSED);
  * не повторяет одно и то же: каждое состояние показывается один раз;
  * пишет всё в CSV-журнал и, если включено, шлёт в Telegram.

Запуск:
    python bybit_radar_scanner.py                     # непрерывный мониторинг
    python bybit_radar_scanner.py --once              # один проход и выход
    python bybit_radar_scanner.py --min-conf 3        # только сильные сигналы
    python bybit_radar_scanner.py --tf 60             # только часовой
    python bybit_radar_scanner.py --category linear   # только TradFi перпы

Зависимости: только стандартная библиотека Python 3.8+.
"""

import argparse
import csv
import json
import math
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

# Windows-консоль по умолчанию не в UTF-8 — иначе русский текст превращается в мусор
if os.name == "nt":
    os.system("")                                   # включает ANSI-цвета
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                               # noqa: BLE001
        pass
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:                               # noqa: BLE001
        pass

BASE = "https://api.bybit.com"
STATE_FILE = "radar_state.json"
VOL_CACHE_FILE = "radar_volatility.json"
LOG_FILE = "radar_signals.csv"
MSK = timezone(timedelta(hours=3))

# ---------------------------------------------------------------- настройки
TELEGRAM_TOKEN = ""   # опционально: токен бота
TELEGRAM_CHAT = ""    # опционально: id чата

TF_NAMES = {"60": "1ч", "240": "4ч"}

# ---------------------------------------------------------------- сеть
def http_get(url, retries=3):
    ctx = ssl.create_default_context()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "radar-scanner/1.0"})
            with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
                return json.load(r)
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"запрос не удался: {url} ({last})")


def get_instruments(categories):
    """Список инструментов: TradFi перпы и xStocks."""
    out = []
    for cat in categories:
        cursor = ""
        while True:
            url = f"{BASE}/v5/market/instruments-info?category={cat}&limit=1000"
            if cursor:
                url += f"&cursor={cursor}"
            data = http_get(url)["result"]
            for item in data.get("list", []):
                if item.get("status") != "Trading":
                    continue
                stype = item.get("symbolType", "")
                if cat == "linear" and stype == "stock":
                    out.append((cat, item["symbol"], "TradFi"))
                elif cat == "spot" and stype == "xstocks":
                    out.append((cat, item["symbol"], "xStock"))
            cursor = data.get("nextPageCursor") or ""
            if not cursor:
                break
    return sorted(set(out))


def get_klines(category, symbol, interval, limit=300):
    """Свечи от старых к новым. Последняя — текущая, ещё не закрытая."""
    url = (f"{BASE}/v5/market/kline?category={category}&symbol={symbol}"
           f"&interval={interval}&limit={limit}")
    rows = http_get(url)["result"]["list"]
    rows = sorted(rows, key=lambda x: int(x[0]))
    return {
        "ts":     [int(x[0]) for x in rows],
        "open":   [float(x[1]) for x in rows],
        "high":   [float(x[2]) for x in rows],
        "low":    [float(x[3]) for x in rows],
        "close":  [float(x[4]) for x in rows],
        "volume": [float(x[5]) for x in rows],
    }


# ---------------------------------------------------------------- волатильность
def daily_volatility(category, symbol, bars=60):
    """Средний дневной ход в процентах (ATR14 / цена) и годовая волатильность."""
    url = (f"{BASE}/v5/market/kline?category={category}&symbol={symbol}"
           f"&interval=D&limit={bars}")
    rows = sorted(http_get(url)["result"]["list"], key=lambda x: int(x[0]))
    # Часть TradFi-контрактов запущена недавно: у них всего 7-25 дневных свечей.
    # Считаем по тому, что есть, и помечаем оценку как приблизительную.
    if len(rows) < 8:
        return None
    h = [float(x[2]) for x in rows]
    l = [float(x[3]) for x in rows]
    c = [float(x[4]) for x in rows]
    tr = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
          for i in range(1, len(c))]
    window = min(14, len(tr))
    atr = sum(tr[-window:]) / window
    if c[-1] <= 0:
        return None
    rets = [c[i] / c[i - 1] - 1 for i in range(1, len(c))][-30:]
    mean = sum(rets) / len(rets)
    sd = (sum((x - mean) ** 2 for x in rets) / len(rets)) ** 0.5
    return {"atr_pct": round(atr / c[-1] * 100, 2),
            "annual_pct": round(sd * (252 ** 0.5) * 100, 1),
            "bars": len(rows),
            "approx": len(rows) < 25}


def load_vol_cache(max_age_hours=12):
    if not os.path.exists(VOL_CACHE_FILE):
        return {}
    try:
        with open(VOL_CACHE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("ver") != 2:
            return {}
        if time.time() - data.get("stamp", 0) > max_age_hours * 3600:
            return {}
        return data.get("items", {})
    except Exception:            # noqa: BLE001
        return {}


def save_vol_cache(items):
    tmp = VOL_CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"stamp": int(time.time()), "ver": 2, "items": items}, fh)
    os.replace(tmp, VOL_CACHE_FILE)


def build_vol_profile(instruments, workers=6, refresh=False):
    """Профиль волатильности по инструментам, с суточным кешем."""
    cache = {} if refresh else load_vol_cache()
    todo = [(c, s) for c, s, _ in instruments if s not in cache]
    if todo:
        print(f"Считаю волатильность: {len(todo)} инструментов "
              f"(кеш на {len(cache)})...", flush=True)

        def one(pair):
            cat, sym = pair
            try:
                return sym, daily_volatility(cat, sym)
            except Exception:    # noqa: BLE001
                return sym, None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for sym, val in pool.map(one, todo):
                cache[sym] = val
        save_vol_cache(cache)
    return cache


# ---------------------------------------------------------------- индикаторы
def rma(values, length):
    out, prev = [], None
    alpha = 1.0 / length
    for v in values:
        prev = v if prev is None else prev + alpha * (v - prev)
        out.append(prev)
    return out


def ema(values, length):
    out, prev = [], None
    alpha = 2.0 / (length + 1)
    for v in values:
        prev = v if prev is None else prev + alpha * (v - prev)
        out.append(prev)
    return out


def sma(values, length):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= length:
            acc -= values[i - length]
        out.append(acc / length if i >= length - 1 else float("nan"))
    return out


def stdev(values, length):
    out = []
    for i in range(len(values)):
        if i < length - 1:
            out.append(float("nan"))
            continue
        w = values[i - length + 1:i + 1]
        m = sum(w) / length
        out.append(math.sqrt(sum((x - m) ** 2 for x in w) / length))
    return out


def rsi(close, length=14):
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(close)):
        d = close[i] - close[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    up, dn = rma(gains, length), rma(losses, length)
    out = []
    for u, d in zip(up, dn):
        out.append(50.0 if d == 0 else 100.0 - 100.0 / (1.0 + u / d))
    return out


def true_range(h, l, c):
    out = [h[0] - l[0]]
    for i in range(1, len(c)):
        out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return out


def cci(h, l, c, length=20):
    tp = [(h[i] + l[i] + c[i]) / 3 for i in range(len(c))]
    ma = sma(tp, length)
    out = []
    for i in range(len(c)):
        if i < length - 1 or math.isnan(ma[i]):
            out.append(float("nan"))
            continue
        dev = sum(abs(tp[j] - ma[i]) for j in range(i - length + 1, i + 1)) / length
        out.append(float("nan") if dev == 0 else (tp[i] - ma[i]) / (0.015 * dev))
    return out


def fisher(h, l, length=9):
    mid = [(h[i] + l[i]) / 2 for i in range(len(h))]
    vs, fs, v, f = [], [], 0.0, 0.0
    for i in range(len(mid)):
        if i < length - 1:
            vs.append(0.0)
            fs.append(0.0)
            continue
        w = mid[i - length + 1:i + 1]
        lo, hi = min(w), max(w)
        raw = 0.0 if hi == lo else 2 * ((mid[i] - lo) / (hi - lo)) - 1
        v = max(-0.999, min(0.999, 0.33 * raw + 0.67 * v))
        f = 0.5 * math.log((1 + v) / (1 - v)) * 0.5 + 0.5 * f
        vs.append(v)
        fs.append(f)
    return fs


def ultimate(h, l, c):
    bp, tr = [0.0], [h[0] - l[0]]
    for i in range(1, len(c)):
        low = min(l[i], c[i - 1])
        bp.append(c[i] - low)
        tr.append(max(h[i], c[i - 1]) - low)

    def avg(n, i):
        if i < n:
            return None
        s_tr = sum(tr[i - n + 1:i + 1])
        return None if s_tr == 0 else sum(bp[i - n + 1:i + 1]) / s_tr

    out = []
    for i in range(len(c)):
        a7, a14, a28 = avg(7, i), avg(14, i), avg(28, i)
        out.append(float("nan") if None in (a7, a14, a28)
                   else 100.0 * (4 * a7 + 2 * a14 + a28) / 7.0)
    return out


def td_setup(close, direction):
    out, cnt = [], 0
    for i in range(len(close)):
        if i < 4:
            out.append(0)
            continue
        cond = close[i] < close[i - 4] if direction == "buy" else close[i] > close[i - 4]
        cnt = cnt + 1 if cond else 0
        out.append(cnt)
    return out


def connors_rsi(close):
    streak, s = [], 0.0
    for i in range(len(close)):
        if i == 0:
            streak.append(0.0)
            continue
        if close[i] > close[i - 1]:
            s = s + 1 if s > 0 else 1
        elif close[i] < close[i - 1]:
            s = s - 1 if s < 0 else -1
        else:
            s = 0
        streak.append(s)
    roc = [0.0] + [(close[i] / close[i - 1] - 1) * 100 if close[i - 1] else 0.0
                   for i in range(1, len(close))]
    rank = []
    for i in range(len(close)):
        if i < 100:
            rank.append(float("nan"))
            continue
        w = roc[i - 100:i]
        rank.append(sum(1 for x in w if x < roc[i]) / len(w) * 100)
    r3, rs2 = rsi(close, 3), rsi(streak, 2)
    return [float("nan") if math.isnan(rank[i]) else (r3[i] + rs2[i] + rank[i]) / 3
            for i in range(len(close))]


# ---------------------------------------------------------------- логика Radar
COMPONENTS_LONG = ["CCI<-200", "нижняя BB", "RSI<30", "Fisher", "UltOsc", "TD9 buy", "ConnorsRSI<10"]
COMPONENTS_TAKE = ["CCI>200", "верхняя BB", "RSI>70", "Fisher", "UltOsc", "TD9 sell", "ConnorsRSI>90"]


def compute(k, window=3):
    """Возвращает состояние на последнем баре: conf вниз/вверх и что сработало."""
    c, h, l = k["close"], k["high"], k["low"]
    n = len(c)
    if n < 120:
        return None

    r = rsi(c, 14)
    cc = cci(h, l, c, 20)
    basis, dev = sma(c, 20), stdev(c, 20)
    fi = fisher(h, l, 9)
    uo = ultimate(h, l, c)
    tdb, tds = td_setup(c, "buy"), td_setup(c, "sell")
    cr = connors_rsi(c)
    e200 = ema(c, 200)

    def cross_dn(series, level, i):
        return (not math.isnan(series[i]) and not math.isnan(series[i - 1])
                and series[i] < level <= series[i - 1])

    def cross_up(series, level, i):
        return (not math.isnan(series[i]) and not math.isnan(series[i - 1])
                and series[i] > level >= series[i - 1])

    def bb_lower_break(i):
        if math.isnan(basis[i]) or math.isnan(dev[i]) or math.isnan(basis[i - 1]):
            return False
        return c[i] < basis[i] - 2 * dev[i] and c[i - 1] >= basis[i - 1] - 2 * dev[i - 1]

    def bb_upper_break(i):
        if math.isnan(basis[i]) or math.isnan(dev[i]) or math.isnan(basis[i - 1]):
            return False
        return c[i] > basis[i] + 2 * dev[i] and c[i - 1] <= basis[i - 1] + 2 * dev[i - 1]

    def fisher_up(i):
        return fi[i] < -1.5 and fi[i] > fi[i - 1] and fi[i - 1] <= fi[i - 2]

    def fisher_dn(i):
        return fi[i] > 1.5 and fi[i] < fi[i - 1] and fi[i - 1] >= fi[i - 2]

    def fired_long(i):
        return [cross_dn(cc, -200, i), bb_lower_break(i), cross_dn(r, 30, i),
                fisher_up(i), cross_up(uo, 30, i), tdb[i] == 9, cross_dn(cr, 10, i)]

    def fired_take(i):
        return [cross_up(cc, 200, i), bb_upper_break(i), cross_up(r, 70, i),
                fisher_dn(i), cross_dn(uo, 70, i), tds[i] == 9, cross_up(cr, 90, i)]

    def confluence(fn, i):
        active = [False] * 7
        for back in range(window):
            j = i - back
            if j < 2:
                break
            for idx, val in enumerate(fn(j)):
                active[idx] = active[idx] or val
        return active

    last = n - 1
    act_long_now = confluence(fired_long, last)
    act_long_prev = confluence(fired_long, last - 1)
    act_take_now = confluence(fired_take, last)
    act_take_prev = confluence(fired_take, last - 1)

    return {
        "ts": k["ts"][last],
        "close": c[last],
        "rsi": r[last],
        "below_ema200": c[last] < e200[last],
        "conf_long": sum(act_long_now),
        "conf_long_prev": sum(act_long_prev),
        "parts_long": [COMPONENTS_LONG[i] for i, v in enumerate(act_long_now) if v],
        "fired_now_long": [COMPONENTS_LONG[i] for i, v in enumerate(fired_long(last)) if v],
        "conf_take": sum(act_take_now),
        "conf_take_prev": sum(act_take_prev),
        "parts_take": [COMPONENTS_TAKE[i] for i, v in enumerate(act_take_now) if v],
        "fired_now_take": [COMPONENTS_TAKE[i] for i, v in enumerate(fired_take(last)) if v],
    }


def bar_is_closed(ts_ms, interval):
    """Свеча с началом ts_ms закрыта, если её конец уже в прошлом."""
    minutes = int(interval)
    end = ts_ms + minutes * 60 * 1000
    return end <= int(time.time() * 1000)


# ---------------------------------------------------------------- уведомления
def notify_console(text, kind):
    colors = {"LONG": "\033[92m", "TAKE": "\033[91m", "INFO": "\033[96m"}
    reset = "\033[0m"
    print(f"{colors.get(kind, '')}{text}{reset}", flush=True)


def notify_sound():
    try:
        import winsound
        winsound.Beep(880, 180)
        winsound.Beep(1320, 180)
    except Exception:            # noqa: BLE001
        pass


def notify_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        payload = json.dumps({"chat_id": TELEGRAM_CHAT, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception:            # noqa: BLE001
        pass


def log_row(row):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        if not exists:
            w.writerow(["время_МСК", "символ", "тип_инструмента", "таймфрейм", "сторона",
                        "совпало", "статус", "цена", "RSI", "ниже_EMA200", "ATR_%", "компоненты"])
        w.writerow(row)


# ---------------------------------------------------------------- состояние
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:        # noqa: BLE001
            return {}
    return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_FILE)


# ---------------------------------------------------------------- основной цикл
def scan_symbol(args, category, symbol, kind, interval):
    try:
        k = get_klines(category, symbol, interval)
        st = compute(k, window=args.window)
        if st is None:
            return []
    except Exception as exc:     # noqa: BLE001
        if args.verbose:
            print(f"  [!] {symbol} {interval}: {exc}", file=sys.stderr)
        return []

    closed = bar_is_closed(st["ts"], interval)
    events = []
    for side, conf, prev, parts, fired in (
        ("LONG", st["conf_long"], st["conf_long_prev"], st["parts_long"], st["fired_now_long"]),
        ("TAKE", st["conf_take"], st["conf_take_prev"], st["parts_take"], st["fired_now_take"]),
    ):
        if side == "TAKE" and args.no_take:
            continue
        if conf < args.min_conf or conf <= prev:
            continue
        events.append({
            "symbol": symbol, "kind": kind, "category": category,
            "interval": interval, "side": side, "conf": conf,
            "closed": closed, "ts": st["ts"], "close": st["close"],
            "rsi": st["rsi"], "below_ema200": st["below_ema200"],
            "parts": parts, "fired": fired,
        })
    return events


def format_event(ev, vol_profile=None):
    tf = TF_NAMES.get(ev["interval"], ev["interval"])
    when = datetime.fromtimestamp(ev["ts"] / 1000, MSK).strftime("%d.%m %H:%M")
    status = "ЗАКРЫТА" if ev["closed"] else "ФОРМИРУЕТСЯ"
    label = "ЛОНГ" if ev["side"] == "LONG" else "ВЕРШИНА / фиксация"
    mark = "" if ev["side"] == "LONG" else (" · ниже EMA200" if ev["below_ema200"] else "")
    info = (vol_profile or {}).get(ev["symbol"])
    if info:
        mark += f" · ATR {info['atr_pct']:g}%" + ("~" if info.get("approx") else "")
    now = datetime.now(MSK).strftime("%H:%M:%S")
    return (f"[{now}] {ev['symbol']:<16} {ev['kind']:<7} {tf:<3} "
            f"{label:<18} {ev['conf']}/7  {status:<11} "
            f"цена {ev['close']:<12.4f} RSI {ev['rsi']:.0f}{mark}\n"
            f"          свеча {when} МСК · сработали: {', '.join(ev['fired']) or 'перенос из окна'}"
            f" · в окне: {', '.join(ev['parts'])}")


def run_once(args, instruments, state, vol_profile=None):
    tasks = []
    for category, symbol, kind in instruments:
        for interval in args.tf:
            tasks.append((category, symbol, kind, interval))

    found = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_symbol, args, c, s, k, i) for c, s, k, i in tasks]
        for fut in futures:
            found.extend(fut.result())

    fresh = []
    for ev in found:
        key = f"{ev['symbol']}|{ev['interval']}|{ev['side']}|{ev['ts']}|{ev['conf']}|{int(ev['closed'])}"
        if key in state:
            continue
        state[key] = int(time.time())
        fresh.append(ev)

    # чистим старые записи состояния, чтобы файл не рос бесконечно
    cutoff = int(time.time()) - 14 * 24 * 3600
    for key in [k for k, v in state.items() if v < cutoff]:
        del state[key]

    for ev in fresh:
        info = (vol_profile or {}).get(ev["symbol"])
        if info:
            ev["atr_pct"] = info["atr_pct"]
    fresh.sort(key=lambda e: (-e["conf"], -e.get("atr_pct", 0), e["symbol"]))
    for ev in fresh:
        text = format_event(ev, vol_profile)
        notify_console(text, ev["side"])
        log_row([datetime.now(MSK).strftime("%Y-%m-%d %H:%M:%S"), ev["symbol"], ev["kind"],
                 TF_NAMES.get(ev["interval"], ev["interval"]), ev["side"], ev["conf"],
                 "закрыта" if ev["closed"] else "формируется",
                 f"{ev['close']:.6f}", f"{ev['rsi']:.1f}",
                 "да" if ev["below_ema200"] else "нет",
                 f"{ev.get('atr_pct', 0):.2f}" if ev.get("atr_pct") else "",
                 ", ".join(ev["parts"])])
        notify_telegram(text)
    if fresh and not args.quiet:
        notify_sound()
    return fresh


def main():
    p = argparse.ArgumentParser(description="Reversal Radar для акций на Bybit")
    p.add_argument("--tf", nargs="+", default=["60", "240"],
                   help="таймфреймы: 60 (1ч), 240 (4ч). По умолчанию оба")
    p.add_argument("--min-conf", type=int, default=2,
                   help="минимум совпавших компонентов (по умолчанию 2)")
    p.add_argument("--window", type=int, default=3, help="окно совпадения в барах")
    p.add_argument("--category", nargs="+", default=["linear", "spot"],
                   help="linear — TradFi перпы, spot — xStocks")
    p.add_argument("--interval", type=int, default=90, help="пауза между проходами, секунд")
    p.add_argument("--workers", type=int, default=6, help="параллельных запросов")
    p.add_argument("--once", action="store_true", help="один проход и выход")
    p.add_argument("--no-take", action="store_true", help="только лонговые сигналы")
    p.add_argument("--quiet", action="store_true", help="без звука")
    p.add_argument("--verbose", action="store_true", help="показывать ошибки запросов")
    p.add_argument("--only", nargs="+", help="ограничить конкретными символами")
    p.add_argument("--min-atr", type=float, default=0.0,
                   help="минимальный средний дневной ход в процентах (ATR14/цена). "
                        "0 — без фильтра. Ориентир: USAR около 6%%, медиана по списку 4.5%%")
    p.add_argument("--volatile", action="store_true",
                   help="ярлык для --min-atr 4 — только заметно волатильные бумаги")
    p.add_argument("--vol-top", type=int, default=0,
                   help="оставить только N самых волатильных инструментов")
    p.add_argument("--refresh-vol", action="store_true",
                   help="пересчитать волатильность, игнорируя суточный кеш")
    args = p.parse_args()

    print("Получаю список инструментов с Bybit...", flush=True)
    instruments = get_instruments(args.category)
    if args.only:
        wanted = {s.upper() for s in args.only}
        instruments = [x for x in instruments if x[1].upper() in wanted]
    total_before = len(instruments)

    min_atr = 4.0 if args.volatile and args.min_atr <= 0 else args.min_atr
    vol_profile = {}
    if min_atr > 0 or args.vol_top > 0:
        vol_profile = build_vol_profile(instruments, workers=args.workers,
                                        refresh=args.refresh_vol)
        scored, no_data = [], []
        for item in instruments:
            info = vol_profile.get(item[1])
            (scored if info else no_data).append((item, info) if info else item)
        if min_atr > 0:
            kept = [(it, v) for it, v in scored if v["atr_pct"] >= min_atr]
            print(f"Фильтр волатильности: ATR >= {min_atr:g}%  —  "
                  f"осталось {len(kept)} из {len(scored)}"
                  + (f", без данных о волатильности пропущено {len(no_data)}" if no_data else ""))
            scored = kept
        scored.sort(key=lambda pair: -pair[1]["atr_pct"])
        if args.vol_top > 0:
            scored = scored[:args.vol_top]
            print(f"Оставлены {len(scored)} самых волатильных")
        instruments = [it for it, _ in scored]
        if instruments:
            top = scored[:5]
            print("Самые волатильные в скане: "
                  + ", ".join(f"{it[1]} {v['atr_pct']:g}%" + ("~" if v.get("approx") else "")
                              for it, v in top))
            approx = sum(1 for _, v in scored if v.get("approx"))
            if approx:
                print(f"  (~ у {approx} контрактов меньше 25 дневных свечей — "
                      f"оценка волатильности приблизительная)")

    tradfi = sum(1 for x in instruments if x[2] == "TradFi")
    xst = sum(1 for x in instruments if x[2] == "xStock")
    print(f"Инструментов в скане: {len(instruments)}  (TradFi перпы: {tradfi}, xStocks: {xst})")
    print(f"Таймфреймы: {', '.join(TF_NAMES.get(t, t) for t in args.tf)}   "
          f"порог: {args.min_conf}/7   пауза: {args.interval} с")
    print("Сигнал показывается сразу на формирующейся свече и отдельно после её закрытия.")
    print(f"Журнал: {os.path.abspath(LOG_FILE)}")
    print("-" * 110, flush=True)

    state = load_state()
    try:
        while True:
            started = time.time()
            fresh = run_once(args, instruments, state, vol_profile)
            save_state(state)
            if not fresh and not args.quiet:
                print(f"[{datetime.now(MSK).strftime('%H:%M:%S')}] тихо · "
                      f"проход занял {time.time() - started:.0f} с", flush=True)
            if args.once:
                break
            time.sleep(max(5, args.interval - (time.time() - started)))
    except KeyboardInterrupt:
        save_state(state)
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
