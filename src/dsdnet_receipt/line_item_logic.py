"""Logic checks on extracted line items.

The item table is read from OCR geometry (`semantic.extract_line_items`); the
scanner often loses a column, repeats one row's numbers on the rows around it,
or leaves the quantity and price inside the description. These checks use the
arithmetic every receipt obeys (quantity x unit price = amount; the items add
up to the total) and change a row only when the numbers printed on the
receipt satisfy that arithmetic. Each changed row says what was worked out.
"""
from __future__ import annotations

import re
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any

from .semantic import LINE_ITEM_KEYS

_CENT = Decimal("0.01")
_UNIT_WORDS = r"PCS|PC|UNIT|UNITS|BAG|BAGS|BOX|BTL|SET|SETS|KG|G|L|LTR|M|ROLL|PKT|PACK|EA|NOS|TIN|CTN"
_QTY_TOKEN = re.compile(rf"(?i)(?<![\d.])(\d{{1,4}}(?:\.\d{{1,3}})?)\s*({_UNIT_WORDS})\b")
_MONEY = re.compile(r"(?<![\d.])(\d{1,6}(?:,\d{3})*\.\d{2})(?![\d.])")
_HEADER = re.compile(r"(?i)^(?:(?:type|descr\w*|desc\w*|atir|attr|size|qty|quantity|cuty|cty|price|ext|sale|item|items|"
                     r"unit|u/price|amount|amt|disc\.?|code|no\.?)\s+)+")
_NOT_PRODUCT = re.compile(r"(?i)\b(?:item\s*sold|items?\s*sold|subtotal|sub\s*total|total|tenders?|payment|cash|change|"
                          r"rounding|thank|please|dear|customer|receipt|tax|sst|gst)\b")
_FUEL_HINT = re.compile(r"(?i)pump|litre|liter|ltr|\w*[lith]tre\b|diesel|d[ie]{1,2}[sz]el|ron\s*9[57]|euro\s*5|petrol|fuel|"
                        r"petronas|petron|shell|caltex|bhp|five\s*star")


def _d(value: Any) -> Decimal | None:
    try:
        text = str(value).replace(",", "").strip()
        return Decimal(text) if text else None
    except (InvalidOperation, ValueError):
        return None


def _note(row: dict[str, str], text: str) -> None:
    notes = [n for n in (row.get("review_note") or "").split("; ") if n and not n.startswith("Verify")]
    if text not in notes:
        notes.append(text)
    row["review_note"] = "; ".join(notes)


def _texts(lines) -> list[str]:
    return [str(line.get("text", "")) if isinstance(line, dict) else str(line) for line in lines]


def _letters(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text or ""))


def _split_embedded(row: dict[str, str]) -> bool:
    """'P08 BM6A PLUG 2.00 PCS 6.00' with amount 12.00 -> qty 2, PCS, 6.00."""
    desc = row.get("description") or ""
    amount = _d(row.get("amount"))
    numbers = [(m.group(1), m.start(), m.end()) for m in re.finditer(r"(?<![\w.])(\d{1,4}(?:\.\d{1,3})?)(?![\w.])", desc)]
    money = [(m.group(1), m.start(), m.end()) for m in re.finditer(r"(?i)(?:RM)?(\d{1,6}\.\d{2})(?![\d.])", desc)]
    if amount is None:
        return False
    for q_text, qs, qe in numbers:
        for p_text, ps, pe in money:
            if ps <= qs < pe or qs <= ps < qe:
                continue
            q, p = _d(q_text), _d(p_text)
            if q and p and q > 0 and (q * p).quantize(_CENT) == amount:
                unit = re.match(rf"(?i)\s*({_UNIT_WORDS})\b", desc[qe:])
                start = min(qs, ps)
                end = max(pe, qe + (unit.end() if unit else 0))
                row["quantity"] = row.get("quantity") or f"{q.normalize():f}"
                row["unit_price"] = row.get("unit_price") or f"{p:.2f}"
                if unit and not row.get("unit"):
                    row["unit"] = unit.group(1).upper()
                left = desc[:start].strip()
                left = re.sub(r"^\d{1,2}\s+(?=\S)", "", left)        # a leading row number
                row["description"] = (left + " " + desc[end:].strip()).strip(" /-") or desc
                _note(row, "quantity and price read from the description; quantity x price = amount")
                return True
    if not row.get("quantity") and not row.get("unit_price"):
        single = re.search(r"(?i)(?<![\w.])1\s+RM\s*(\d{1,6}\.\d{2})$", desc)
        if single and _d(single.group(1)) == amount:
            row["quantity"], row["unit_price"] = "1", f"{amount:.2f}"
            row["description"] = desc[:single.start()].strip()
            _note(row, "quantity 1 at the printed price")
            return True
    return False


def _complete_from_text(row: dict[str, str], texts: list[str], money: list[Decimal]) -> bool:
    """Unit price known, amount or quantity lost: a printed quantity q and a
    printed amount a with q x price = a."""
    price = _d(row.get("unit_price"))
    if price is None or price <= 0 or row.get("amount"):
        return False
    quantities = []
    for text in texts:
        for value, unit in _QTY_TOKEN.findall(text):
            q = _d(value)
            if q and q > 0:
                quantities.append((q, unit.upper()))
    if row.get("quantity"):
        q = _d(row["quantity"])
        if q and (q * price).quantize(_CENT) in money:
            row["amount"] = f"{(q * price).quantize(_CENT):.2f}"
            _note(row, "amount = quantity x price, as printed")
            return True
        return False
    for q, unit in quantities:
        a = (q * price).quantize(_CENT)
        if a in money and q != 1:
            row["quantity"], row["amount"] = f"{q.normalize():f}", f"{a:.2f}"
            row["unit"] = row.get("unit") or unit
            _note(row, "quantity and amount found on the receipt; quantity x price = amount")
            return True
    if price in money and any(q == 1 for q, _ in quantities):
        unit = next(u for q, u in quantities if q == 1)
        row["quantity"], row["amount"], row["unit"] = "1", f"{price:.2f}", row.get("unit") or unit
        _note(row, "quantity 1 printed; amount = price")
        return True
    return False


def _code_like(description: str) -> bool:
    """A pump, terminal or model code rather than a product (POS1, PBL#, MODELM-39816")."""
    d = (description or "").strip()
    if _letters(d) == 0:
        return True
    if "#" in d:
        return True
    return " " not in d and bool(re.search(r"\d", d))


def _drop_repeats(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Neighbouring rows that carry exactly the same numbers were read from one
    printed row: keep the row whose description is a product, fold codes into it."""
    groups: list[list[dict[str, str]]] = []
    for row in rows:
        numbers = (row.get("unit_price"), row.get("amount"))
        if groups and any(numbers) and numbers == (groups[-1][0].get("unit_price"), groups[-1][0].get("amount")):
            groups[-1].append(row)
        else:
            groups.append([row])
    kept: list[dict[str, str]] = []
    for group in groups:
        products = [r for r in group if not _code_like(r.get("description"))]
        codes = [r for r in group if _code_like(r.get("description"))]
        if len(group) == 1 or not codes or not products:
            kept.extend(group)
            continue
        for product in products:
            kept.append(product)
        if len(products) == 1 and any(re.search(r"(?i)model|s/?no|serial|[\"]", c.get("description", "")) for c in codes):
            products[0]["description"] += " / " + " / ".join(c["description"] for c in codes)
    for index, row in enumerate(kept, 1):
        row["item"] = str(index)
    return kept


def _product_name_below(row: dict[str, str], texts: list[str]) -> None:
    """A description that is only headers and codes: the product name printed
    on its own line under the price (Glitter Heart Swim Ring)."""
    desc = _HEADER.sub("", row.get("description") or "").strip()
    if _letters(desc) >= 1 and not re.search(r"\d{8,}", desc):
        row["description"] = desc
        return
    amount = row.get("amount")
    anchor = next((i for i, t in enumerate(texts) if amount and amount in t.replace(",", "")), None)
    if anchor is None:
        row["description"] = desc or row.get("description", "")
        return
    for text in texts[anchor + 1: anchor + 5]:
        if _letters(text) >= 2 and not _NOT_PRODUCT.search(text) and not re.search(r"\d", text):
            row["description"] = text.strip()
            _note(row, "product name read from the line under the price")
            return
    row["description"] = desc or row.get("description", "")


def _fuel_row(texts: list[str], total: Decimal, product: str | None) -> dict[str, str] | None:
    """Litres x price per litre = total (13.96 x 2.150 = 30.01 -> 30.00)."""
    joined = " ".join(texts)
    if not _FUEL_HINT.search(joined):
        return None
    litres, prices = set(), set()
    for token in re.findall(r"\d{3,6}(?:\.\d{1,3})?|\d{1,3}\.\d{1,3}", joined):
        if "." in token:
            value = _d(token)
            if value is None:
                continue
            if Decimal("1.5") <= value <= Decimal("4.5") and len(token.split(".")[1]) == 3:
                prices.add(value)
            if Decimal("1") <= value <= Decimal("200"):
                litres.add(value)
        else:
            if len(token) == 4:
                prices.add(Decimal(token) / 1000)
            if 3 <= len(token) <= 5:
                litres.add(Decimal(token) / 100)
    prices = {p for p in prices if Decimal("1.5") <= p <= Decimal("4.5")}
    found = []
    for litre in litres:
        for price in prices:
            if abs(litre * price - total) <= Decimal("0.05") and litre != total:
                found.append((litre, price))
    found = sorted({(l.quantize(_CENT, rounding=ROUND_DOWN), p) for l, p in found})
    if len(found) != 1:
        return None
    litre, price = found[0]
    if not product:
        product = "Diesel" if re.search(r"(?i)diesel|d[ie]{1,2}[sz]el|deus", joined) else "Fuel"
    row = {key: "" for key in LINE_ITEM_KEYS}
    row.update(item="1", description=product, quantity=f"{litre:.2f}", unit="L", unit_price=f"{price:.3f}", amount=f"{total:.2f}")
    _note(row, "litres x price per litre = total")
    return row


def refine_line_items(items: list[dict[str, str]], lines, total: str | None, fuel_product: str | None = None) -> list[dict[str, str]]:
    texts = _texts(lines)
    money = []
    for text in texts:
        money += [_d(m) for m in _MONEY.findall(text)]
    money = [m for m in money if m is not None]
    rows = [dict(row) for row in items or []]
    for row in rows:
        _split_embedded(row)
        _complete_from_text(row, texts, money)
        if not row.get("quantity") and row.get("unit_price") and row.get("amount"):
            p, a = _d(row["unit_price"]), _d(row["amount"])
            if p and a and p > 0 and (a / p) == (a / p).to_integral_value() and 1 <= a / p <= 999:
                row["quantity"] = f"{(a / p).to_integral_value()}"
                _note(row, "quantity = amount / price")
        if row.get("quantity") and row.get("unit_price") and not row.get("amount"):
            q, p = _d(row["quantity"]), _d(row["unit_price"])
            if q and p:
                row["amount"] = f"{(q * p).quantize(_CENT):.2f}"
                _note(row, "amount worked out as quantity x price; check")
        _product_name_below(row, texts)
    rows = _drop_repeats(rows)
    total_value = _d(total)
    # one row repeating another's amount while the items overshoot the total:
    # that row's amount is the remainder
    if total_value is not None and len(rows) >= 2:
        amounts = [_d(r.get("amount")) for r in rows]
        if all(a is not None for a in amounts) and sum(amounts) != total_value:
            for i in range(1, len(rows)):
                if amounts[i] == amounts[i - 1]:
                    rest = total_value - (sum(amounts) - amounts[i])
                    if rest > 0:
                        rows[i]["amount"] = f"{rest:.2f}"
                        if rows[i].get("unit_price") == f"{amounts[i]:.2f}":
                            rows[i]["unit_price"] = ""
                        _note(rows[i], "amount = receipt total less the other items (the scanner repeated the row above); check")
                        break
    # fuel: the litres and price per litre on a pump receipt
    if total_value is not None and total_value > 0:
        no_detail = not rows or (len(rows) == 1 and not rows[0].get("quantity"))
        if no_detail:
            fuel = _fuel_row(texts, total_value, fuel_product)
            if fuel:
                if rows and _letters(rows[0].get("description")) and not _code_like(rows[0]["description"]):
                    fuel["description"] = rows[0]["description"] if not fuel_product else fuel_product
                rows = [fuel]
    amounts = [_d(r.get("amount")) for r in rows]
    if rows and total_value is not None and all(a is not None for a in amounts) and sum(amounts) == total_value:
        for row in rows:
            _note(row, "items add up to the receipt total")
    for row in rows:
        missing = [label for key, label in (("quantity", "quantity"), ("unit_price", "unit price"), ("amount", "amount")) if not row.get(key)]
        if missing:
            _note(row, "Verify " + ", ".join(missing))
    return rows
