"""Telugu text frontend: Unicode normalisation, digit/number verbalisation, grapheme tokenisation.

Telugu orthography is almost phonemic, so (as is common for Indic TTS) every Unicode code point
(consonant, vowel sign, virama, anusvara, ...) is one acoustic token and MFA is trained from scratch with a
grapheme dictionary (no pretrained Telugu MFA model / G2P is needed). ZWJ/ZWNJ are dropped.
"""
import re
import unicodedata

ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
        "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
# combining forms used when a number is followed by a hundred/thousand (ఒక వంద -> వంద, రెండు వందల ...)
PUNCT = set(",.;:?!।")
PUNCT_SP = set(",;:.?!।")


def _below100(n):
    if n < 20:
        return [ONES[n]]
    t, o = divmod(n, 10)
    return [TENS[t]] + ([ONES[o]] if o else [])


def number_to_words(n):
    """Telugu cardinal for 0 <= n < 100000 (space separated words)."""
    if n < 100:
        return " ".join(_below100(n))
    out = []
    th, rest = divmod(n, 1000)
    if th:
        out += (["వెయ్యి"] if th == 1 else _below100(th) + ["వేల"]) if th < 100 else [number_to_words(th), "వేల"]
    h, rest = divmod(rest, 100)
    if h:
        out += ["వంద"] if h == 1 else [ONES[h], "వందల"]
    if rest:
        out += _below100(rest)
    return " ".join(out)


def ordinal(n):
    w = number_to_words(n).split()
    last = w[-1]
    # stem + వ  (పన్నెండు -> పన్నెండవ, ఒకటి -> ఒకటవ)
    if last.endswith("ు") or last.endswith("ి"):
        last = last[:-1]
    w[-1] = last + "వ"
    return " ".join(w)


def normalize(text, keep_punct=False):
    t = unicodedata.normalize("NFC", text)
    t = t.replace("‌", "").replace("‍", "")
    # time 10:45 -> "పది నలభై ఐదు"
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: number_to_words(int(m.group(1))) + (
        "" if int(m.group(2)) == 0 else " " + number_to_words(int(m.group(2)))), t)
    # ordinals 12వ
    t = re.sub(r"(\d+)\s*వ\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), t)
    pat = r"[^ఀ-౿ ,.;:?!]" if keep_punct else r"[^ఀ-౿ ]"
    t = re.sub(pat, " ", t)
    return re.sub(r"\s+", " ", t).strip()


def text_to_symbols(text):
    """Telugu text -> list of grapheme symbols; punctuation becomes the pause symbol 'sp'.

    Word boundaries are not marked (MFA only inserts sil/sp where an actual pause is aligned).
    """
    out = []
    for ch in normalize(text, keep_punct=True):
        if ch == " ":
            continue
        if ch in PUNCT_SP:
            if out and out[-1] != "sp":
                out.append("sp")
        else:
            out.append(ch)
    return out


def word_pron(word):
    return list(normalize(word))
