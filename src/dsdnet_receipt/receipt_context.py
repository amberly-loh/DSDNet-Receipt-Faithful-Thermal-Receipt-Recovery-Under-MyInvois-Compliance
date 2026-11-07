"""What a receipt is for, read from its own words.

The OCR text of a degraded receipt is often glued together and slightly
misspelt ("PSBatu7Jatontpoh", "Erterprise", "Pnmax95"). This module:

* repairs words only against a fixed lexicon of receipt, Malaysian address and
  business words (a word is changed only when one lexicon word is clearly the
  closest; otherwise it is left as read);
* recognises the business: known brands, the kind of shop, and what was bought
  (fuel grade, groceries, meals ...), with the words that gave it away;
* separates a franchise brand (PETRONAS) from the business that issued the
  receipt (the dealer's "... Enterprise" / "... Sdn Bhd" line under it).

Everything returned is labelled as recognised from the receipt text, never as
read verbatim, and the user can change it.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

# --- lexicon -----------------------------------------------------------------

ADDRESS_WORDS = (
    "Jalan", "Jln", "Lorong", "Persiaran", "Lebuhraya", "Lebuh", "Taman", "Tmn", "Kampung", "Kg", "Bandar",
    "Batu", "Lot", "Mukim", "Daerah", "Seksyen", "Section", "Medan", "Pusat", "Kompleks", "Plaza", "Wisma",
    "Menara", "Bangunan", "Kawasan", "Perindustrian", "Industri", "Perdagangan", "Komersial", "Desa", "Bukit",
    "Sungai", "Simpang", "Pekan", "Lapangan", "Tingkat", "Aras", "Ground", "Floor", "Block", "Blok", "Unit",
    "Station", "Stesen", "Utama", "Baru", "Lama", "Tengah", "Barat", "Timur", "Utara", "Selatan",
)
PLACES = (
    "Kuala Lumpur", "Selangor", "Putrajaya", "Johor", "Johor Bahru", "Kedah", "Kelantan", "Melaka", "Malacca",
    "Negeri Sembilan", "Pahang", "Penang", "Pulau Pinang", "Perak", "Perlis", "Sabah", "Sarawak", "Terengganu",
    "Labuan", "Ipoh", "Klang", "Shah Alam", "Petaling Jaya", "Subang Jaya", "Puchong", "Cheras", "Kajang",
    "Seremban", "Ampang", "Gombak", "Rawang", "Sentul", "Kepong", "Setapak", "Selayang", "Cyberjaya", "Bangi",
    "Semenyih", "Seri Kembangan", "Sri Petaling", "Bukit Jalil", "Damansara", "Bangsar", "Cheras", "Kuantan",
    "Kota Bharu", "Kota Kinabalu", "Kuching", "Miri", "Sibu", "Alor Setar", "Sungai Petani", "Georgetown",
    "Butterworth", "Bayan Lepas", "Taiping", "Seri Iskandar", "Batu Pahat", "Muar", "Kluang", "Skudai", "Nilai",
    "Port Dickson", "Kuala Terengganu", "Kangar", "Temerloh", "Bentong", "Genting", "Wangsa Maju", "Setiawangsa",
)
BUSINESS_WORDS = (
    "Enterprise", "Enterprises", "Trading", "Sdn", "Bhd", "Berhad", "Resources", "Holdings", "Services",
    "Industries", "Marketing", "Supplies", "Restoran", "Restaurant", "Kedai", "Runcit", "Pasaraya", "Mini",
    "Market", "Supermarket", "Hypermarket", "Station", "Petrol", "Bakery", "Cafe", "Farmasi", "Pharmacy", "Klinik",
    "Clinic", "Hardware", "Motor", "Auto", "Tyre", "Workshop", "Stationery", "Bookstore", "Kopitiam",
    "Dagangan", "Perniagaan", "Niaga", "Usaha", "Jaya", "Maju", "Sentosa", "Makmur", "Bersatu", "Malaysia",
    # Malay number words are common in business names
    "Satu", "Dua", "Tiga", "Empat", "Lima", "Enam", "Tujuh", "Lapan", "Sembilan", "Sepuluh", "Puluh", "Ratus",
)
RECEIPT_WORDS = (
    "Receipt", "Invoice", "Tax", "Cash", "Sale", "Sales", "Total", "Grand", "Subtotal", "Rounding", "Change",
    "Cashier", "Pump", "Date", "Time", "Copy", "Final", "Official", "Customer", "Member", "Points", "Qty",
    "Amount", "Price", "Discount", "Payment", "Card", "Credit", "Debit", "Litre", "Liter", "Litres",
)

# brand -> (business type, expense type key)
BRANDS = {
    "PETRONAS": ("Petrol station", "fuel"), "SHELL": ("Petrol station", "fuel"), "PETRON": ("Petrol station", "fuel"),
    "CALTEX": ("Petrol station", "fuel"), "BHPETROL": ("Petrol station", "fuel"), "BH PETROL": ("Petrol station", "fuel"),
    "MESRA": ("Petrol station shop", "fuel"), "SETEL": ("Petrol station", "fuel"),
    "LOTUS'S": ("Supermarket", "inventory"), "LOTUSS": ("Supermarket", "inventory"), "TESCO": ("Supermarket", "inventory"),
    "GIANT": ("Supermarket", "inventory"), "MYDIN": ("Supermarket", "inventory"), "AEON": ("Supermarket", "inventory"),
    "JAYA GROCER": ("Supermarket", "inventory"), "VILLAGE GROCER": ("Supermarket", "inventory"),
    "NSK": ("Supermarket", "inventory"), "SPEEDMART": ("Grocery / convenience store", "inventory"),
    "99 SPEEDMART": ("Grocery / convenience store", "inventory"), "ECONSAVE": ("Supermarket", "inventory"),
    "HERO": ("Supermarket", "inventory"), "BILLION": ("Supermarket", "inventory"),
    "7-ELEVEN": ("Convenience store", "meals"), "7ELEVEN": ("Convenience store", "meals"),
    "FAMILYMART": ("Convenience store", "meals"), "FAMILY MART": ("Convenience store", "meals"),
    "KK MART": ("Convenience store", "meals"), "KKMART": ("Convenience store", "meals"),
    "MR DIY": ("Hardware / household store", "maintenance"), "MR.DIY": ("Hardware / household store", "maintenance"),
    "ACE HARDWARE": ("Hardware store", "maintenance"), "DAISO": ("Household store", "office"),
    "POPULAR": ("Bookstore / stationery", "office"), "MPH": ("Bookstore", "office"),
    "WATSONS": ("Pharmacy / personal care", "medical"), "GUARDIAN": ("Pharmacy / personal care", "medical"),
    "CARING": ("Pharmacy", "medical"), "BIG PHARMACY": ("Pharmacy", "medical"),
    "MCDONALD": ("Restaurant / fast food", "meals"), "KFC": ("Restaurant / fast food", "meals"),
    "STARBUCKS": ("Cafe", "meals"), "TEALIVE": ("Cafe / beverages", "meals"), "ZUS": ("Cafe", "meals"),
    "OLDTOWN": ("Cafe / restaurant", "meals"), "SECRET RECIPE": ("Cafe / restaurant", "meals"),
    "PIZZA HUT": ("Restaurant / fast food", "meals"), "DOMINO": ("Restaurant / fast food", "meals"),
    "SUBWAY": ("Restaurant / fast food", "meals"), "MARRYBROWN": ("Restaurant / fast food", "meals"),
    "TNB": ("Electricity", "utilities"), "TENAGA": ("Electricity", "utilities"), "UNIFI": ("Internet / telco", "utilities"),
    "TELEKOM": ("Internet / telco", "utilities"), "MAXIS": ("Telco", "utilities"), "CELCOM": ("Telco", "utilities"),
    "DIGI": ("Telco", "utilities"), "INDAH WATER": ("Sewerage", "utilities"), "AIR SELANGOR": ("Water", "utilities"),
    "TOUCH 'N GO": ("Toll / transport", "fuel"), "TOUCHNGO": ("Toll / transport", "fuel"), "PLUS": ("Toll", "fuel"),
    "GRAB": ("Transport / delivery", "fuel"), "AIRASIA": ("Travel", "travel"), "MALAYSIA AIRLINES": ("Travel", "travel"),
    "CEX": ("Second-hand electronics & games", "other"), "SENHENG ": ("Electrical store", "office"),
    "MACHINES": ("Electronics store", "office"), "SWITCH": ("Electronics store", "office"),
    "IKEA": ("Furniture / household", "office"), "COURTS": ("Electrical / furniture", "office"),
    "SENHENG": ("Electrical store", "office"), "HARVEY NORMAN": ("Electrical store", "office"),
}
FUEL_PRODUCTS = (
    (r"P[A-Z0-9]{0,2}MAX\s*97|RON\s*97|BLAZE\s*97|V-?POWER\s*97", "RON97 petrol"),
    (r"P[A-Z0-9]{0,2}MAX\s*95|PR[I1L]MAX|RON\s*95|BLAZE\s*95|FUEL\s*SAVE\s*95|V-?POWER\s*95", "RON95 petrol"),
    (r"V-?POWER\s*RACING|RON\s*100", "RON100 petrol"),
    (r"DIESEL|EURO\s*5\s*B?\d*|DYNAMIC\s*DIESEL", "Diesel"),
)
GOODS_KEYWORDS = {
    "groceries": ("rice", "beras", "sugar", "gula", "milk", "susu", "egg", "telur", "bread", "roti", "flour", "tepung",
                  "oil", "minyak masak", "vegetable", "sayur", "chicken", "ayam", "fish", "ikan", "meat", "daging",
                  "biscuit", "noodle", "mee", "coffee", "kopi", "tea", "teh", "water", "air mineral", "detergent",
                  "tissue", "shampoo", "soap", "sabun"),
    "meals": ("nasi", "mee goreng", "teh tarik", "set", "meal", "burger", "latte", "ais", "bungkus", "makan",
              "chicken rice", "roti canai"),
    "office": ("paper", "a4", "pen", "stapler", "file", "folder", "toner", "ink", "printer", "envelope", "stationery"),
    "maintenance": ("screw", "paint", "cat", "pipe", "paip", "cement", "simen", "bolt", "nut", "drill", "bulb",
                    "cable", "wayar", "tape", "glue", "spanner", "repair", "service"),
    "medical": ("panadol", "paracetamol", "medicine", "ubat", "vitamin", "mask", "plaster", "consultation"),
}

_LEXICON = tuple(sorted({w.upper() for w in ADDRESS_WORDS + BUSINESS_WORDS + RECEIPT_WORDS}
                        | {"PRIMAX", "DIESEL", "ROUNDING", "MALAYSIA"}
                        | {part.upper() for place in PLACES for part in place.split()}, key=len, reverse=True))
_LEXICON_SET = set(_LEXICON)
# brand names are kept exactly as printed, never used as a correction target
_PROTECTED = _LEXICON_SET | {b for b in BRANDS if b.isalpha()}
# characters a thermal-print reader commonly swaps
_SHAPES = str.maketrans({"1": "I", "L": "I", "|": "I", "0": "O", "5": "S", "8": "B", "6": "G", "2": "Z"})


@lru_cache(maxsize=400_000)
def _similar(token: str, word: str) -> float:
    a, b = token.upper(), word.upper()
    plain = SequenceMatcher(None, a, b).ratio()
    shaped = SequenceMatcher(None, a.translate(_SHAPES), b.translate(_SHAPES)).ratio()
    return max(plain, shaped)


@lru_cache(maxsize=50_000)
def correct_word(token: str, prefer: frozenset = frozenset()) -> str:
    """The lexicon word a token was clearly meant to be, else the token itself.
    Short tokens, numbers and codes are never changed."""
    letters = sum(ch.isalpha() for ch in token)
    if letters < 4 or letters < len(token) - 1 or token.upper() in _PROTECTED:
        return token
    # short words need a closer match unless the neighbouring words are of the
    # same kind ("tapan Puluh Tujuh" -> Lapan); "Black" must not become "Block"
    need = 0.85 if len(token) <= 5 else 0.75
    loose = 0.8 if len(token) <= 5 else 0.75
    scored = sorted(((_similar(token, word), word) for word in _LEXICON if abs(len(word) - len(token)) <= 2),
                    reverse=True)
    if not scored or scored[0][0] < loose:
        return token
    tied = [w for score, w in scored if score >= scored[0][0] - 0.02]
    if len(set(tied)) > 1:
        # two words fit equally well: the neighbours' kind of word decides, else leave as read
        preferred = [w for w in tied if w in prefer]
        if len(set(preferred)) != 1:
            return token
        word = preferred[0]
    else:
        word = scored[0][1]
    if scored[0][0] < need and word not in prefer:
        return token
    return word if token.isupper() else word.title()


def _break_glued(run: str) -> list[str]:
    """Split a glued run of letters into lexicon words where an exact or near
    split exists ("KualaLumpur", "JalanIpoh"); else return it whole."""
    upper = run.upper()
    n = len(upper)
    if n < 7 or upper in _PROTECTED:
        return [run]
    best: list[tuple[float, list[tuple[int, int]]] | None] = [None] * (n + 1)
    best[0] = (0.0, [])
    for end in range(1, n + 1):
        for start in range(max(0, end - 14), end):
            if best[start] is None:
                continue
            piece = upper[start:end]
            if len(piece) >= 3 and (piece in _LEXICON_SET or _near_lexicon(piece) >= 0.8):
                gain = len(piece) * (1.0 if piece in _LEXICON_SET else 0.8)
            else:
                gain = -1.0 * len(piece)  # unknown characters cost
            total = best[start][0] + gain
            if best[end] is None or total > best[end][0]:
                best[end] = (total, best[start][1] + [(start, end)])
    score, spans = best[n]
    known = sum(e - s for s, e in spans if e - s >= 3 and (upper[s:e] in _LEXICON_SET or gain_ok(upper[s:e])))
    if len(spans) < 2 or known < n:  # every piece must be a word; no stray letters left over
        return [run]
    return [run[s:e] for s, e in spans]


@lru_cache(maxsize=100_000)
def _near_lexicon(piece: str) -> float:
    """Best similarity of a piece to any lexicon word of about the same length (cached:
    the same pieces recur across lines and views, and each check is costly)."""
    return max((_similar(piece, w) for w in _LEXICON if abs(len(w) - len(piece)) <= 1), default=0)


def gain_ok(piece: str) -> bool:
    return _near_lexicon(piece) >= 0.8


def tidy_text(text: str) -> str:
    """Restore spacing and repair lexicon words in one OCR line."""
    return _tidy_text(str(text or ""))


@lru_cache(maxsize=50_000)
def _tidy_text(value: str) -> str:
    value = re.sub(r"(?<=[A-Za-z])[()](?=[a-z])", "", value)        # stray bracket inside a word: Lun)pur
    value = re.sub(r"\b[0-9](?=[a-z]{2,}(?![a-z]))", lambda m: m.group(0).translate(_SHAPES), value)  # 8atu -> Batu
    value = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]{2})", " ", value)  # BatuJalan, PSBatu
    value = re.sub(r"(?<=[A-Za-z]{2})(?=\d)|(?<=\d)(?=[A-Za-z]{2})", " ", value)  # Batu7Jalan
    value = re.sub(r"(?<=\w)\(", " (", value)
    value = re.sub(r"\)(?=\w)", ") ", value)
    tokens = value.split()
    groups = {"number": {w.upper() for w in BUSINESS_WORDS[-12:]}, "business": {w.upper() for w in BUSINESS_WORDS[:-12]},
              "address": {w.upper() for w in ADDRESS_WORDS} | {p.upper() for pl in PLACES for p in pl.split()}}
    present = [name for name, members in groups.items() if any(t.upper().strip("(),.") in members for t in tokens)]
    prefer = frozenset().union(*(groups[name] for name in present)) if present else frozenset()
    words = []
    for index, token in enumerate(tokens):
        core = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", token)
        lead, body, tail = core.groups()
        pieces = _break_glued(body) if body.isalpha() else [body]
        neighbours = {t.upper().strip("(),.") for t in tokens[max(0, index - 1):index] + tokens[index + 1:index + 2]}
        words.append(lead + " ".join(correct_word(p, prefer - neighbours) for p in pieces) + tail)
    value = " ".join(words)
    value = re.sub(r"\bP\s*/?\s*S\b", "P/S", value)  # petrol station "P/S"
    value = re.sub(r"\bP[A-Za-z0-9]{0,2}max(?=\s*9[57]\b)", "Primax", value, flags=re.I)  # PETRONAS fuel grade
    for place in PLACES:  # rejoin and repair two-word place names read as one
        pattern = r"\b" + r"\s*".join(re.escape(part) for part in place.split()) + r"\b"
        value = re.sub(pattern, lambda m: place.upper() if m.group(0).isupper() else place, value, flags=re.I)
    return re.sub(r"\s{2,}", " ", value).strip()


# --- recognising the business -----------------------------------------------

_TRANSACTION_LINE = re.compile(
    r"(?i)\b(?:date|time|tarikh|masa|receipt|resit|invoice|copy|cashier|kasyer|pump|pam|terminal|pos|v\s*no|"
    r"trans(?:action)?|txn|member|qty|total|amount|cash|change|rounding)\b|\bRM\s*\d|\d+\.\d{2}\b|"
    r"PR[I1L]MAX|RON\s*9|DIESEL")
_OPERATOR = re.compile(r"(?i)\b(?:enterprise|enterprises|trading|sdn\.?\s*bhd\.?|berhad|resources|holdings|services|"
                       r"industries|marketing|supplies|station)\b")


def _brand_in(text: str, merchant: str | None = None) -> tuple[str, str, str] | None:
    """The brand that issued the receipt. The merchant name is checked first,
    and a brand that only appears as a shopping-centre name in the address
    ("LOT F06, AEON Mall Rawang", "Mydin Mall") does not count."""
    if merchant:
        # the brand named first ("CeX AEON Rawang" is a CeX shop in AEON)
        name = " " + re.sub(r"[^A-Z0-9' ]", " ", str(merchant).upper()) + " "
        hits = []
        for brand, (kind, expense) in BRANDS.items():
            spot = name.find(" " + brand.strip() + " ")
            if spot >= 0:
                hits.append((spot, -len(brand), brand, kind, expense))
        if hits:
            _, _, brand, kind, expense = min(hits)
            return brand.strip(), kind, expense
        found = _brand_in(str(merchant))
        if found:
            return found
    text = re.sub(r"(?i)\b[A-Z'][A-Z'&.]*\s+(?:MALL|MEGAMALL|PLAZA|CENTRE|CENTER|SQUARE)\b", " ", text)
    upper = re.sub(r"\s+", " ", text.upper())
    squashed = re.sub(r"[^A-Z0-9']", "", upper)
    for brand, (kind, expense) in BRANDS.items():
        key = re.sub(r"[^A-Z0-9']", "", brand)
        if len(key) <= 4:
            if re.search(rf"(?<![A-Z0-9]){re.escape(brand)}(?![A-Z0-9])", upper):
                return brand, kind, expense
        elif key in squashed:
            return brand, kind, expense
    # a lightly misread brand name on its own line ("PETR0NAS", "SHEL L")
    for line in upper.splitlines():
        token = re.sub(r"[^A-Z0-9']", "", line)
        if 5 <= len(token) <= 14:
            for brand, (kind, expense) in BRANDS.items():
                key = re.sub(r"[^A-Z0-9']", "", brand)
                if len(key) >= 5 and _similar(token, key) >= 0.85:
                    return brand, kind, expense
    return None


def recognise_receipt(text: str, lines: list[dict[str, Any]] | None = None, merchant: str | None = None,
                      items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Business type, expense type, operator and purchase summary, each with the
    words that support it. Empty values when the receipt does not say."""
    text = str(text or "")
    ordered = [str(line.get("text", "")) for line in (lines or [])] or text.splitlines()
    joined = "\n".join(ordered + [str(merchant or "")])
    squashed = re.sub(r"\s+", "", (joined + "\n" + text).upper())
    clues: list[str] = []
    brand = _brand_in("\n".join(ordered), merchant)
    business_type = expense = None
    if brand:
        clues.append(f"brand {brand[0]}")
        business_type, expense = brand[1], brand[2]

    fuel = next((label for pattern, label in FUEL_PRODUCTS if re.search(pattern, squashed)), None)
    pump = re.search(r"PUMP\s*(?:NO\.?)?\s*[:#]?\s*(\d{1,2})|PAM\s*(\d{1,2})", joined, re.I)
    litres = re.search(r"(\d{1,3}(?:\.\d{1,3})?)\s*(?:L|LTR|LITRE|LITER)S?\b", joined, re.I)
    if fuel or pump:
        clues.append(" / ".join(filter(None, (fuel, f"pump {next(g for g in pump.groups() if g)}" if pump else None))))
        if not business_type:
            business_type, expense = "Petrol station", "fuel"

    item_text = " ".join(str(row.get("description") or row.get("item") or "") for row in items or []).lower()
    goods = item_text or joined.lower()
    goods_hits = {kind: [w for w in words if re.search(rf"\b{re.escape(w)}\b", goods)] for kind, words in GOODS_KEYWORDS.items()}
    goods_kind = max(goods_hits, key=lambda k: len(goods_hits[k]))
    if len(goods_hits[goods_kind]) >= 2:
        clues.append("items: " + ", ".join(goods_hits[goods_kind][:4]))
        if not business_type:
            business_type = {"groceries": "Grocery / supermarket", "meals": "Restaurant / food", "office": "Stationery / office supplies",
                             "maintenance": "Hardware", "medical": "Pharmacy / clinic"}[goods_kind]
            expense = {"groceries": "inventory"}.get(goods_kind, goods_kind)
    if not business_type:
        for word, kind, key in (("restoran", "Restaurant / food", "meals"), ("restaurant", "Restaurant / food", "meals"),
                                ("kopitiam", "Restaurant / food", "meals"), ("cafe", "Cafe", "meals"),
                                ("bakery", "Bakery", "meals"), ("farmasi", "Pharmacy", "medical"),
                                ("pharmacy", "Pharmacy", "medical"), ("klinik", "Clinic", "medical"),
                                ("clinic", "Clinic", "medical"), ("hardware", "Hardware", "maintenance"),
                                ("stationery", "Stationery / office supplies", "office"), ("runcit", "Grocery store", "inventory"),
                                ("pasaraya", "Supermarket", "inventory"), ("mini market", "Grocery store", "inventory"),
                                ("workshop", "Vehicle workshop", "maintenance"), ("tyre", "Tyre / vehicle service", "maintenance"),
                                ("hotel", "Hotel", "travel"), ("parking", "Parking", "fuel"), ("toll", "Toll", "fuel")):
            if re.search(rf"\b{word}\b", joined, re.I):
                business_type, expense = kind, key
                clues.append(f"word '{word}'")
                break

    # the business that issued the receipt: a "... Enterprise / Sdn Bhd" line near the top
    operator = None
    for raw in ordered[:8]:
        line = tidy_text(raw)
        if _OPERATOR.search(line) and not _TRANSACTION_LINE.search(line) and sum(c.isalpha() for c in line) >= 6:
            operator = re.sub(r"\s*\(.*?\)\s*$", "", line).strip(" .,:")
            break

    summary = None
    if fuel or (business_type or "").startswith("Petrol"):
        summary = "Fuel" + (f" ({fuel}" + (f", {litres.group(1)} L" if litres else "") + ")" if fuel else "")
    elif len(goods_hits[goods_kind]) >= 2:
        summary = {"groceries": "Groceries", "meals": "Food & drinks", "office": "Office supplies",
                   "maintenance": "Hardware / repair items", "medical": "Medicine / health items"}[goods_kind]
    elif business_type:
        summary = business_type
    return {
        "brand": brand[0] if brand else None,
        "business_type": business_type,
        "expense_type": expense,
        "operator": operator,
        "purchase_summary": summary,
        "fuel_product": fuel,
        "clues": [c for c in clues if c],
    }


def clean_address_line(text: str) -> str | None:
    """An address line with spacing and words repaired; the part from the first
    transaction word (DATE, PUMP, V NO, amounts ...) onward is dropped."""
    line = tidy_text(text)
    cut = _TRANSACTION_LINE.search(line)
    if cut:
        line = line[:cut.start()].strip(" ,.;:-")
    line = line.strip()
    return line if sum(ch.isalnum() for ch in line) >= 4 else None


def is_transaction_line(text: str) -> bool:
    return bool(_TRANSACTION_LINE.search(tidy_text(text)))


# --- reading copy -------------------------------------------------------------

_KEEP_SHORT = {"RM", "SST", "GST", "QTY", "TEL", "FAX", "NO", "PCS", "KG", "UNIT", "CASH", "CARD", "TAX", "SR", "ZR"}


def _box(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(p[0]) for p in line["box"]]; ys = [float(p[1]) for p in line["box"]]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a, b) -> float:
    """Shared area as a share of the smaller box."""
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(small, 1.0)


def _line_score(line: dict[str, Any]) -> float:
    text = str(line.get("text", ""))
    readable = sum(ch.isalnum() or ch in " ./&'-():" for ch in text) / max(len(text), 1)
    known = sum(word.upper().strip("(),.:") in _LEXICON_SET for word in tidy_text(text).split())
    return float(line.get("confidence") or 0) + 0.05 * readable + 0.04 * known + 0.002 * min(len(text), 40)


def reading_copy_lines(lines: list[dict[str, Any]], *, tidy: bool = True) -> list[dict[str, Any]]:
    """One tidy reading per printed place on the receipt. The scanner reads the
    receipt several times (different cleaned views), so the same words come back
    two or three times at almost the same spot; only the best reading of each
    spot is kept, then spacing and known words are repaired. Fragments that are
    not words (two or three stray letters) are dropped."""
    candidates = [(line, _box(line)) for line in lines or [] if str(line.get("text", "")).strip()]
    candidates = [(line, box) for line, box in candidates if box]
    candidates.sort(key=lambda item: _line_score(item[0]), reverse=True)
    kept: list[tuple[dict[str, Any], tuple]] = []
    for line, box in candidates:
        if any(_overlap(box, other) > 0.45 for _, other in kept):
            continue
        kept.append((line, box))
    out = []
    for line, box in kept:
        text = _total_label(tidy_text(str(line.get("text", "")))) if tidy else str(line.get("text", "")).strip()
        letters = re.sub(r"[^A-Za-z]", "", text)
        if not re.search(r"\d", text) and len(letters) <= 3 and letters.upper() not in _KEEP_SHORT | _LEXICON_SET:
            continue
        if float(line.get("confidence") or 0) < 0.55:
            continue
        out.append({"text": text, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    return _reading_order(out)


def _reading_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-to-bottom, and left-to-right within a printed row. Pieces of one row
    ("Cash" ... "10.00") rarely share exactly the same height, so rows are formed
    from vertical overlap rather than from the exact centre of each box."""
    rows: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda item: (item["y0"] + item["y1"]) / 2):
        height = max(item["y1"] - item["y0"], 1)
        if rows:
            top = min(other["y0"] for other in rows[-1])
            bottom = max(other["y1"] for other in rows[-1])
            row_height = max(min(other["y1"] - other["y0"] for other in rows[-1]), 1)
            shared = min(bottom, item["y1"]) - max(top, item["y0"])
            if shared >= 0.5 * min(height, row_height):
                rows[-1].append(item)
                continue
        rows.append([item])
    return [item for row in rows for item in sorted(row, key=lambda item: item["x0"])]


_TOTAL_LABELS = ("SUBTOTAL", "ROUNDING", "TOTAL", "GRAND TOTAL", "CHANGE", "CASH", "TAX", "DISCOUNT", "BALANCE")


def _total_label(text: str) -> str:
    """A lone, badly read word in the totals block ("sthtoral", "Rounagg") is
    shown as the totals label it clearly is, when one label stands out."""
    word = text.strip()
    if " " in word or not word.isalpha() or len(word) < 5 or word.upper() in _TOTAL_LABELS:
        return text
    scored = sorted(((_similar(word, label.replace(" ", "")), label) for label in _TOTAL_LABELS), reverse=True)
    if scored[0][0] >= 0.6 and scored[0][0] - scored[1][0] >= 0.12:
        return scored[0][1].title()
    return text
