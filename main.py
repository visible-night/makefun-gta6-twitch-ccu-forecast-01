"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

Model: launch-day shape matching against past big launches.

Every comparable in the case file is a past game launch with a full day of
hourly Twitch category viewers. For each one we measure the shape of its
launch day as log-ratios to its own first hour: how much each hour grew or
fell relative to hour 1. The forecast anchors on this case's hour-1 print
and applies the comparables' shape to it:

  * Comparables are weighted by how close their own hour-1 audience is to
    this case's hour-1 print (a Gaussian kernel on the log scale). A launch
    that opened to a similar-sized crowd is a better guide to the day's
    shape than one that opened tiny or enormous; this also down-weights
    launches whose hour 1 was still pre-unlock, since their hour-1 counts
    sit far from a big launch's.
  * For each hour, the forecast's log-median is the weighted median of the
    comparables' log-ratios, lightly smoothed across neighbouring hours.
  * The spread is the weighted interquartile range of the same log-ratios
    (converted to a normal sigma), floored so the forecast never claims
    more certainty than the data supports. Quantiles come from the normal
    approximation around the median.

Everything is computed from the case file at run time: no network, no
clock, no randomness, so two runs produce the same bytes. Standard library
only; runs in well under a second per case.

Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input).
Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output), one JSON object
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]} for hours 2..24.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Gaussian kernel width (log viewers) for weighting comparables by how
# close their hour-1 audience is to this case's hour-1 print.
SCALE_BAND = 1.0
# Spread floor and cap (log units) for an hour's normal sigma.
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.0
# Smallest number of comparables needed to trust an hour's own spread;
# thinner hours borrow from their neighbours.
MIN_PER_HOUR = 3
# Output bounds the competition enforces on every quantile.
Q_MIN = 1.0
Q_MAX = 1_000_000_000.0

_NORMAL = NormalDist()


def _parse_time(text):
    """An ISO 8601 UTC timestamp to a datetime (handles the trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _round2(value):
    """Round to 2 decimals, halves away from zero, as a plain number."""
    scaled = math.floor(abs(value) * 100 + 0.5) / 100.0
    return -scaled if value < 0 else scaled


def _weighted_quantile(pairs, tau):
    """Weighted tau-quantile of [(value, weight)] with linear interpolation."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    run = 0.0
    prev_value, prev_run = ordered[0][0], 0.0
    for value, weight in ordered:
        run += weight
        if run >= target:
            if run == prev_run:
                return value
            frac = (target - prev_run) / (run - prev_run)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_run = value, run
    return ordered[-1][0]


def _launch_day_hourly(comparable):
    """Map hour (1..24) -> viewers for a comparable's release day (UTC)."""
    day_start = _parse_time(comparable["day_start"])
    hours = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            when = _parse_time(point["from"])
        except (KeyError, ValueError):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = viewers
    return hours


def _shape_series(case, hour_one):
    """One (weight, {hour: log-ratio to hour 1}) per usable comparable."""
    log_hour_one = math.log(max(1.0, hour_one))
    series = []
    for comparable in case.get("comparables") or []:
        try:
            hours = _launch_day_hourly(comparable)
        except (KeyError, ValueError):
            continue
        first = hours.get(1)
        if not first or first <= 0:
            continue
        gap = (math.log(first) - log_hour_one) / SCALE_BAND
        weight = math.exp(-0.5 * gap * gap)
        ratios = {}
        for hour in range(2, 25):
            viewers = hours.get(hour)
            if viewers and viewers > 0:
                ratios[hour] = math.log(viewers / first)
        if ratios:
            series.append((weight, ratios))
    return series


def _hour_estimates(series):
    """Per hour: (log-median, sigma) from the comparables' log-ratios.

    Hours with too few comparables are filled by linear interpolation
    between their neighbours (or the nearest hour at the edges).
    """
    median = {}
    sigma = {}
    for hour in range(2, 25):
        pairs = [(ratios[hour], w) for w, ratios in series if hour in ratios]
        if len(pairs) < MIN_PER_HOUR:
            continue
        median[hour] = _weighted_quantile(pairs, 0.5)
        lower = _weighted_quantile(pairs, 0.25)
        upper = _weighted_quantile(pairs, 0.75)
        spread = (upper - lower) / 1.349
        sigma[hour] = min(SIGMA_CAP, max(SIGMA_FLOOR, spread))
    known = sorted(median)
    if not known:
        return {h: 0.0 for h in range(2, 25)}, {h: SIGMA_FLOOR for h in range(2, 25)}
    for hour in range(2, 25):
        if hour in median:
            continue
        before = [h for h in known if h < hour]
        after = [h for h in known if h > hour]
        if before and after:
            lo, hi = before[-1], after[0]
            frac = (hour - lo) / (hi - lo)
            median[hour] = median[lo] + frac * (median[hi] - median[lo])
            sigma[hour] = sigma[lo] + frac * (sigma[hi] - sigma[lo])
        elif before:
            median[hour], sigma[hour] = median[before[-1]], sigma[before[-1]]
        else:
            median[hour], sigma[hour] = median[after[0]], sigma[after[0]]
    return median, sigma


def _smooth(values):
    """Three-hour moving average of the log-median curve (edges: 2 hours)."""
    out = {}
    for hour in range(2, 25):
        near = [values[h] for h in (hour - 1, hour, hour + 1) if 2 <= h <= 24]
        out[hour] = sum(near) / len(near)
    return out


def _hour_one_print(case):
    """This case's hour-1 viewer count, with defensive fallbacks."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers"):
            return max(1.0, float(row["viewers"]))
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last:
            return max(1.0, float(last))
    return 1.0


def _quantiles(levels, median, sigma, anchor):
    """Five quantiles around anchor * exp(median) with log-spread sigma."""
    row = []
    for tau in levels:
        z = _NORMAL.inv_cdf(tau)
        value = anchor * math.exp(median + z * sigma)
        value = min(Q_MAX, max(Q_MIN, value))
        row.append(_round2(value))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def forecast_case(case):
    """Forecast one case: {"hours": [{"hour": h, "q": [...]}]} for h 2..24."""
    levels = tuple(case.get("levels") or LEVELS)
    anchor = _hour_one_print(case)
    series = _shape_series(case, anchor)
    median, sigma = _hour_estimates(series)
    median = _smooth(median)
    wanted = {row["hour"] for row in case.get("forecast") or [] if 2 <= row["hour"] <= 24}
    if len(wanted) != 23:
        wanted = set(range(2, 25))
    hours = [
        {"hour": h, "q": _quantiles(levels, median[h], sigma[h], anchor)}
        for h in sorted(wanted)
    ]
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
