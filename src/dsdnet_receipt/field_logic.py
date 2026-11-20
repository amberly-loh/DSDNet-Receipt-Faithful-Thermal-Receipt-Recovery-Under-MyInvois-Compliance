"""Logic checks on the key fields after extraction and review.

The scanner can lose a decimal point ("RM 30.00" read as 3000), a separator
("25/12/2018" read as 25/122018), or turn one digit into a look-alike
("30" read as 39). Each check below repairs a value only when the receipt's
own content supports exactly one reading, and every repair is recorded with
the earlier value and the reason, as in `field_review`.

* total, missing decimal point: an amount read without a decimal point is
  taken as ringgit and sen (3000 -> 30.00) when the receipt shows that
  reading elsewhere (another line prints 30.00, or cash minus change equals
  it), or when no amount on the receipt kept its decimal point and the
  amount is printed at least twice (total and cash) on a till receipt (a
  printed time); a handwritten bill in whole ringgit (340) is left as it is.
* total, not found: the only "RM" amount on the receipt (RM-10 -> 10.00), or
  cash paid minus change given.
* total, contradicted: when cash paid minus change gives a different amount
  that is also printed on the receipt, that amount is the total (a digit of
  the total was misread, 83.35 for 63.35); not when the total equals the cash
  paid, or when no change was given and the cash reads like the total with a
  digit missing (6.00 for 56.00: then the cash is the misread).
* total, paid: when the amount paid (less any change) is also printed as a
  total or printed twice, and the total read disagrees, the paid amount is
  the total (labels like TOTAI, CA8H, TATAL read on faded print).
* date, not found or impossible: separators restored (25/122018), letters
  read for digits (O->0, I/l->1), a misspelt month (21MR 2018),
  or one look-alike digit (2023-04-39 -> 2023-04-30) when only one valid
  date results.
* date, year unreadable: the day and month are kept as a hint, and a year is
  suggested (never filled) only when exactly one recent year fits the digits
  that were read; the user accepts or corrects it.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
# a digit OCR commonly reads for another on thermal print
_LOOKALIKE = {"0": "89", "9": "0", "8": "03", "3": "8", "6": "5", "5": "6", "1": "7", "7": "1"}
_LETTER_DIGIT = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})


def _texts(lines: list[dict[str, Any]] | list[str]) -> list[str]:
    return [str(line.get("text", "")) if isinstance(line, dict) else str(line) for line in lines]


def _ok(y: int, m: int, d: int, today: date) -> date | None:
    """A real calendar date within the last 15 years (a repaired date older
    than that is more likely a misread than an old receipt)."""
    if y < 100:
        y += 2000
    try:
        value = date(y, m, d)
    except ValueError:
        return None
    return value if date(today.year - 15, 1, 1) <= value <= today + timedelta(days=1) else None


def _digits_for_letters(text: str) -> str:
    """Letters next to digits read as the digit they resemble (2O18 -> 2018)."""
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"(?<=\d)[OoIl|]|[OoIl|](?=\d)", lambda m: m.group(0).translate(_LETTER_DIGIT), text)
    return text


def _one_lookalike(day: int, month: int, year: int, today: date) -> list[date]:
    """Valid dates reachable by changing one digit of the day or month."""
    found = set()
    for which in ("d", "m"):
        digits = f"{day if which == 'd' else month:02d}"
        for i, ch in enumerate(digits):
            for alt in _LOOKALIKE.get(ch, ""):
                n = int(digits[:i] + alt + digits[i + 1:])
                value = _ok(year, month, n, today) if which == "d" else _ok(year, n, day, today)
                if value:
                    found.add(value)
    return sorted(found)


def _month_from(word: str) -> int | None:
    """A month name read with one letter wrong or missing (MR, MSY)."""
    word = word.upper()[:4]
    if word[:3] in _MONTHS:
        return _MONTHS.index(word[:3]) + 1
    near = []
    for index, name in enumerate(_MONTHS, 1):
        w = word[:3] if len(word) >= 3 else word
        if len(w) == 3 and sum(a != b for a, b in zip(w, name)) == 1:
            near.append(index)
        elif len(w) == 2 and (w == name[0] + name[2] or w == name[:2] or w == name[1:]):
            near.append(index)
    return near[0] if len(near) == 1 else None


_TIME_AFTER = r"(?=\D|\d{1,2}\s*[:.]\s*\d{2}|$)"
_DMY_LOOSE = re.compile(r"(?<![\d.])(\d{2})\s*[/\-]\s*(\d{2})\s*[/.\-]?\s*((?:20)\d{2})" + _TIME_AFTER)
_YMD_LOOSE = re.compile(r"(?<!\d)(20\d{2})\s*[/.\-]\s*(\d{2})\s*[/.\-]?\s*(\d{2})" + _TIME_AFTER)
_D_MON_Y = re.compile(r"(?i)(?<!\d)(\d{1,2})\s*([A-Z]{2,4})\s*[-/,]?\s*(20\d{2})" + _TIME_AFTER.replace("(?=\\D", "(?=\\D|\\d{1,4}(?!\\d)"))
# day and month run together before the year: 0202/2017, 2109/2017
_DDMM_Y = re.compile(r"(?<!\d)(\d{2})(\d{2})\s*[/\-]\s*(20\d{2})" + _TIME_AFTER)
_DATE_LABEL = re.compile(r"(?i)\bdat[e3]\b")


def repair_date(lines, today: date | None = None) -> tuple[str | None, str | None, int | None]:
    """A date the stricter patterns missed. Returns (iso date, reason, line index)."""
    today = today or date.today()
    candidates: list[tuple[int, int, date, str]] = []   # (priority, line index, date, reason)
    for index, raw in enumerate(_texts(lines)):
        text = _digits_for_letters(raw)
        labelled = bool(_DATE_LABEL.search(raw))
        found: list[tuple[date, str]] = []
        for d, m, y in _DMY_LOOSE.findall(text):
            value = _ok(int(y), int(m), int(d), today)
            if value:
                found.append((value, "separator restored"))
            else:
                alts = _one_lookalike(int(d), int(m), int(y), today)
                if len(alts) == 1:
                    found.append((alts[0], f"day/month {d}/{m} cannot exist; only one look-alike digit gives a valid date"))
        for y, m, d in _YMD_LOOSE.findall(text):
            value = _ok(int(y), int(m), int(d), today)
            if value:
                found.append((value, "separator restored"))
            else:
                alts = _one_lookalike(int(d), int(m), int(y), today)
                if len(alts) == 1:
                    found.append((alts[0], f"{y}-{m}-{d} cannot exist; only one look-alike digit gives a valid date"))
        for d, m, y in _DDMM_Y.findall(text):
            value = _ok(int(y), int(m), int(d), today)
            if value:
                found.append((value, "separator between day and month restored"))
        for d, word, y in _D_MON_Y.findall(text):
            month = _month_from(word)
            value = _ok(int(y), month, int(d), today) if month else None
            if value:
                found.append((value, "month name read with a letter wrong"))
        for value, why in found:
            candidates.append((0 if labelled else 1, index, value, why))
    if not candidates:
        return None, None, None
    best = min(candidates, key=lambda c: (c[0], c[1]))
    rivals = {c[2] for c in candidates if c[0] == best[0]}
    if len(rivals) > 1 and best[0] == 1:
        return None, None, None   # different unlabelled dates: do not choose
    return best[2].isoformat(), best[3], best[1]


_DM_PARTIAL = re.compile(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{2})(\d{0,6})")


def suggest_partial_date(lines, today: date | None = None) -> tuple[str | None, str | None]:
    """(hint "dd/mm", suggested iso date) when the day and month were read but not the year."""
    today = today or date.today()
    best_hint, best = None, None
    for raw in _texts(lines):
        if not _DATE_LABEL.search(raw) and not re.search(r"(?i)\d\s*[:.]\s*\d{2}\s*(?:[ap]m)?|[ap]m\b", raw):
            continue
        text = _digits_for_letters(raw)
        for d, m, rest in _DM_PARTIAL.findall(text):
            day, month = int(d), int(m)
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue
            hint = f"{day:02d}/{month:02d}"
            window = (m + rest)[:4]
            years = {y for y in range(today.year - 6, today.year + 1)
                     if f"{y % 100:02d}" in window and _ok(y, month, day, today)}
            best_hint = best_hint or hint
            if len(years) == 1:
                return hint, _ok(years.pop(), month, day, today).isoformat()
    return best_hint, best


# ------------------------------------------------------------------ amounts
_DEC = re.compile(r"(?<![\d.])(\d{1,6}(?:,\d{3})*\.\d{2})(?![\d.])")
_RM_ANY = re.compile(r"(?i)\bRM\s*[-:]?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)(?![\d.,])")
_CASH = re.compile(r"(?i)\b(?:cash|tunai|received|tendered|paid)\b\W{0,12}(?:RM\s*)?(\d{1,6}(?:\.\d{2})?)(?![\d.])")
_CHANGE = re.compile(r"(?i)\b(?:change|baki)\b\W{0,12}(?:RM\s*)?(\d{1,6}(?:\.\d{2})?)(?![\d.])")
_TIMEISH = re.compile(r"\d{1,2}[:.]\d{2}[:.]\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}")


def _dec(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _decimal_amounts(texts: list[str]) -> list[Decimal]:
    found = []
    for text in texts:
        for match in _DEC.findall(_TIMEISH.sub(" ", text)):
            value = _dec(match)
            if value is not None and value > 0:
                found.append(value)
    return found


def _as_sen(token: str) -> Decimal | None:
    """A token read without its decimal point, as ringgit and sen (3000 -> 30.00)."""
    if "." in token:
        return _dec(token)
    value = _dec(token)
    return value / 100 if value is not None and len(token) >= 3 else value


def _cash_and_change(texts: list[str]) -> tuple[Decimal | None, Decimal | None]:
    joined = "\n".join(texts)
    cash = _CASH.findall(joined)
    change = _CHANGE.findall(joined)
    if not cash or not change:
        return None, None
    return _as_sen(cash[-1]), _as_sen(change[-1])


def _cash_less_change(texts: list[str]) -> Decimal | None:
    c, g = _cash_and_change(texts)
    if c is None or g is None or c <= g:
        return None
    return c - g


def _dropped_digits(short: Decimal | None, long: Decimal) -> bool:
    """`short` reads like `long` with digits missing (6.00 for 56.00)."""
    if short is None:
        return False
    a, b = f"{short:.2f}".replace(".", ""), f"{long:.2f}".replace(".", "")
    it = iter(b)
    return len(a) < len(b) and all(ch in it for ch in a)


# payment and change labels as the scanner reads them on faded print
_PAY = re.compile(r"(?i)(?<![A-Z])(?:CA[S5$8]H|C[O0]SH|CASH\.|RECE?I?[VY](?:ED|PD|D)|REORD|PAID|TENDER(?:ED)?|PAYMENT|"
                  r"AMOUNT\s*TO\s*BE\s*PAID|VISA|MASTER\s*CARD|DEBIT|CREDIT)\b\W{0,6}(?:RM|\$)?\s*-?\s*(\d{1,6}\.\d{2})(?![\d.])")
_GIVEN = re.compile(r"(?i)(?<![A-Z])(?:CHA?N?R?GE|CHERGE|CHAGE|BAKI|BALANCE)\b\W{0,6}(?:RM|\$)?\s*-?\s*(\d{1,6}\.\d{2})(?![\d.])")
_TOTALISH = re.compile(r"(?i)(?:T[O0]?[TA]?[AI1]?L|TATAL|GRAN[DT]?\s*T?[O0]A?L|INCLU\w*\s*(?:OF\s*)?G?S?T|PAID|PAYABLE|NET\s*AMT|AMOUNT)\W{0,8}(?:RM|\$)?\s*(\d{1,6}\.\d{2})(?![\d.])")


def _paid_total(texts: list[str]) -> Decimal | None:
    """The amount the customer settled: paid when no change was given, else
    paid minus change."""
    joined = "\n".join(_TIMEISH.sub(" ", t) for t in texts)
    paid = [_dec(m) for m in _PAY.findall(joined)]
    given = [_dec(m) for m in _GIVEN.findall(joined)]
    paid = [p for p in paid if p is not None and p > 0]
    if not paid:
        return None
    p = paid[0]
    g = given[0] if given else None
    if g is None or g == 0:
        return p
    return p - g if p > g else None


def _payment_check(value: str | None, texts: list[str], decimals: list[Decimal]) -> tuple[str | None, str | None]:
    candidate = _paid_total(texts)
    if candidate is None:
        return None, None
    current = _dec(value) if value else None
    if current == candidate:
        return None, None
    joined = "\n".join(texts)
    labelled = candidate in {_dec(m) for m in _TOTALISH.findall(joined)}
    repeated = decimals.count(candidate) >= 2
    change_read = bool(_GIVEN.search(joined))
    if not (labelled or repeated or (current is None and change_read)):
        return None, None
    if current is not None and not labelled and decimals.count(current) > decimals.count(candidate):
        return None, None
    return f"{candidate:.2f}", "equals the amount paid (and any change), which the receipt also prints as a total"


def check_total(value: str | None, lines) -> tuple[str | None, str | None]:
    new, why = _check_total_core(value, lines)
    if new:
        return new, why
    texts = _texts(lines)
    return _payment_check(value, texts, _decimal_amounts(texts))


def _check_total_core(value: str | None, lines) -> tuple[str | None, str | None]:
    """(new total, reason) when logic changes or supplies the total, else (None, None)."""
    texts = _texts(lines)
    decimals = _decimal_amounts(texts)
    if value:
        total = _dec(value)
        if total is None:
            return None, None
        # the total disagrees with cash paid minus change, and the receipt
        # prints that amount: a digit of the total was misread (83.35 for 63.35)
        cash, given = _cash_and_change(texts)
        cash_less_change = _cash_less_change(texts)
        if (cash_less_change is not None and cash_less_change != total and cash_less_change in decimals
                and cash != total and not (given == 0 and _dropped_digits(cash, total))):
            return f"{cash_less_change:.2f}", f"cash paid minus change is {cash_less_change:.2f}, and the receipt prints that amount"
        if total < 100 or total != total.to_integral_value():
            return None, None
        digits = f"{int(total)}"
        printed_with_point = any(re.search(rf"(?<![\d.]){re.escape(f'{total:,.2f}')}|(?<![\d.,]){digits}\.\d{{2}}", t) for t in texts)
        read_without_point = any(re.search(rf"(?<!\d){digits}(?![\d.])", _TIMEISH.sub(" ", t)) for t in texts)
        if printed_with_point or not read_without_point:
            return None, None
        alt = total / 100
        if alt in decimals:
            return f"{alt:.2f}", f"read as {digits} without its decimal point; the receipt prints {alt:.2f} elsewhere"
        cash_less_change = _cash_less_change(texts)
        if cash_less_change is not None and cash_less_change == alt:
            return f"{alt:.2f}", f"read as {digits} without its decimal point; cash minus change is {alt:.2f}"
        repeats = sum(len(re.findall(rf"(?<!\d){digits}(?![\d.])", _TIMEISH.sub(" ", t))) for t in texts)
        printed_time = any(re.search(r"\d\s*:\s*[0-5]\d", t) for t in texts)
        if not decimals and repeats >= 2 and printed_time:
            return f"{alt:.2f}", f"read as {digits} (printed {repeats} times, e.g. total and cash); no amount on this receipt kept its decimal point, so it is RM {alt:.2f}"
        return None, None
    # an RM amount without sen counts only when it stands alone on its line
    # ("RM-10"), not inside an item line ("1.25L RM3")
    rm = {_dec(m) for t in texts for m in _RM_ANY.findall(_TIMEISH.sub(" ", t))
          if "." in m or re.fullmatch(r"(?i)\s*RM\s*[-:]?\s*\d{1,6}\s*", t)}
    rm = {v for v in rm if v is not None and v > 0}
    if len(rm) == 1:
        only = rm.pop()
        return f"{only:.2f}", "the only RM amount printed on the receipt"
    cash_less_change = _cash_less_change(texts)
    if cash_less_change is not None and cash_less_change in decimals:
        return f"{cash_less_change:.2f}", "cash paid minus change given"
    return None, None


def apply_logic(lines, extracted: dict[str, dict[str, Any]], today: date | None = None) -> dict[str, Any]:
    """Run the checks on `extracted` in place; returns the changes for the audit."""
    texts = _texts(lines)
    changes: dict[str, Any] = {}

    def change(name, value, reason, index=None):
        field = extracted.setdefault(name, {})
        previous = field.get("value")
        if previous == value:
            return
        field["value"] = value
        field["reviewed_by_app"] = reason
        if index is not None and isinstance(lines[index], dict):
            field["confidence"] = lines[index].get("confidence")
            field["box"] = lines[index].get("box")
        changes[name] = {"from": previous, "to": value, "reason": reason}

    total = (extracted.get("total_amount") or {}).get("value")
    new_total, why = check_total(total, texts)
    if new_total:
        change("total_amount", new_total, why)

    if not (extracted.get("transaction_date") or {}).get("value"):
        repaired, why, index = repair_date(lines, today)
        if repaired:
            change("transaction_date", repaired, why, index)
        else:
            hint, suggested = suggest_partial_date(lines, today)
            field = extracted.setdefault("transaction_date", {})
            if hint and not field.get("hint"):
                field["hint"] = hint
            if suggested and not field.get("suggested"):
                field["suggested"] = suggested
                changes["transaction_date_suggestion"] = {"hint": hint, "suggested": suggested,
                                                          "reason": "only one recent year fits the digits read; the user accepts or corrects it"}
    return {"method": "field_logic_v1", "changes": changes}


# ------------------------------------------------------------------ options
def _date_candidates(lines, today: date, current: str | None = None) -> list[tuple[str, str]]:
    """Every date the receipt supports, best first, with the reason."""
    from .field_review import _dates_in
    out: list[tuple[str, str]] = []

    def add(value: str | None, why: str) -> None:
        if value and value not in [v for v, _ in out]:
            out.append((value, why))

    texts = _texts(lines)
    labelled = [t for t in texts if _DATE_LABEL.search(t)]
    for text in labelled + texts:
        for value in _dates_in(text, today):
            if value >= date(today.year - 15, 1, 1):
                add(value.isoformat(), "printed on the receipt")
    repaired, why, _ = repair_date(lines, today)
    add(repaired, why or "repaired reading")
    if out or current:
        return out
    # day and month read, year not: every recent year that fits
    for raw in texts:
        if not _DATE_LABEL.search(raw) and not re.search(r"(?i)\d\s*[:.]\s*\d{2}|[ap]m\b", raw):
            continue
        text = _digits_for_letters(raw)
        for d, m, rest in _DM_PARTIAL.findall(text):
            day, month = int(d), int(m)
            if not (1 <= day <= 31 and 1 <= month <= 12):
                continue
            window = (m + rest)[:4]
            fits = [y for y in range(today.year, today.year - 7, -1) if f"{y % 100:02d}" in window and _ok(y, month, day, today)]
            for y in fits:
                add(_ok(y, month, day, today).isoformat(), f"day and month read as {day:02d}/{month:02d}; the year digits read fit {y}")
            if not fits:
                for y in [y for y in range(today.year, today.year - 4, -1) if _ok(y, month, day, today)][:2]:
                    add(_ok(y, month, day, today).isoformat(), f"day and month read as {day:02d}/{month:02d}; year unreadable, a recent year")
    return out


def _total_candidates(value: str | None, lines) -> list[tuple[str, str]]:
    texts = _texts(lines)
    joined = "\n".join(texts)
    decimals = _decimal_amounts(texts)
    out: list[tuple[str, str]] = []

    def add(amount, why: str) -> None:
        if amount is None:
            return
        text = f"{amount:.2f}" if isinstance(amount, Decimal) else amount
        if _dec(text) and _dec(text) > 0 and text not in [v for v, _ in out]:
            out.append((text, why))

    new, why = check_total(value, lines)
    add(new, why or "")
    add(_paid_total(texts), "the amount paid (less any change)")
    for match in _TOTALISH.findall(joined):
        add(_dec(match), "printed after a total label")
    printed_time = any(re.search(r"\d\s*:\s*[0-5]\d", t) for t in texts)
    if value and printed_time:
        total = _dec(value)
        if total is not None and total >= 100 and total == total.to_integral_value():
            add(total / 100, f"{value} read without its decimal point")
    return out


def _merchant_candidates(lines, current: str | None = None) -> list[tuple[str, str]]:
    """Company or brand names near the top, and other spellings of the name read."""
    from difflib import SequenceMatcher
    company = re.compile(r"(?i)\b(?:sdn\.?\s*bhd|bhd|enterprise|trading|store|stores|mart|restaurant|cafe|stesen|"
                         r"pharmacy|hardware|supermarket|bakery|clinic)\b|petron|petronas|bhpetrol|shell|caltex|mr\.?\s*d\.?\s*i\.?\s*y")
    noise = re.compile(r"(?i)@|tel|fax|email|jalan|jln|\blot\b|level|customer|cash|date|invoice|receipt|division|no\.|\d{5}")
    key = lambda t: re.sub(r"[^A-Z0-9]", "", (t or "").upper())
    out: list[tuple[str, str]] = []
    seen = {key(current)}
    for text in _texts(lines)[:16]:
        clean = re.sub(r"\s+", " ", text).strip(" .,:-")
        k = key(clean)
        if not k or k in seen or noise.search(clean) or len(re.findall(r"[A-Za-z]{2,}", clean)) < 1:
            continue
        similar = bool(current) and SequenceMatcher(None, k, key(current)).ratio() >= 0.7
        if company.search(clean) or similar:
            seen.add(k)
            out.append((clean, "another reading of the name" if similar else "company or brand name printed at the top"))
        if len(out) >= 3:
            break
    return out


def add_options(lines, extracted: dict[str, dict[str, Any]], today: date | None = None) -> dict[str, Any]:
    """Fill an empty date, total or merchant with the best estimate (marked as
    an estimate) and list the other likely values for the user to pick."""
    today = today or date.today()
    filled: dict[str, Any] = {}
    sources = {
        "transaction_date": _date_candidates(lines, today, (extracted.get("transaction_date") or {}).get("value")),
        "total_amount": _total_candidates((extracted.get("total_amount") or {}).get("value"), lines),
        "merchant": _merchant_candidates(lines, (extracted.get("merchant") or {}).get("value")),
    }
    for name, candidates in sources.items():
        field = extracted.setdefault(name, {})
        current = field.get("value")
        if not current and field.get("suggested"):
            candidates = [(field["suggested"], "year partly hidden; only one recent year fits the digits read")] + \
                         [c for c in candidates if c[0] != field["suggested"]]
        if not current and candidates:
            value, why = candidates[0]
            field["value"] = value
            field["estimated"] = f"Estimated: {why}. Check against the receipt."
            filled[name] = {"to": value, "reason": why}
            current = value
        field["options"] = [{"value": v, "reason": w} for v, w in candidates if v != current][:4]
    return filled
