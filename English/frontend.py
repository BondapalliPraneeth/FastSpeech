"""English text frontend: normalisation (numbers, times, dates) + CMUdict/g2p_en -> stress-marked ARPAbet.

MFA's pretrained english_us_arpa model uses stress-marked vowels (AH0/AH1/AH2) and plain consonants, so the same
symbols are used for the alignment dictionary, the model vocabulary and inference.
"""
import re

from num2words import num2words

_g2p = None
PUNCT_PAUSE = {",": "sp", ";": "sp", ":": "sp", ".": "sp", "?": "sp", "!": "sp"}
CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
ARPA = CONSONANTS + [v + str(i) for v in VOWELS for i in range(3)]


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    s = num2words(h)
    s += " o'clock" if mi == 0 else (" oh " + num2words(mi) if mi < 10 else " " + num2words(mi))
    return s


def normalize(text, keep_punct=False):
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"(\d{1,2}):(\d{2})", _time, t)
    t = re.sub(r"\b(a)\.?m\.?\b", "A M", t, flags=re.I)
    t = re.sub(r"\b(p)\.?m\.?\b", "P M", t, flags=re.I)
    t = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", lambda m: num2words(int(m.group(1)), to="year"), t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), t)
    t = t.replace("-", " ")
    pat = r"[^A-Za-z' ,.;:?!]" if keep_punct else r"[^A-Za-z' ]"
    t = re.sub(pat, " ", t)
    return re.sub(r"\s+", " ", t).strip()


# POS-dependent homographs that CMUdict/g2p_en do not disambiguate: word -> (noun/other, verb) pronunciation
HOMOGRAPHS = {
    "conduct": ("K AA1 N D AH0 K T", "K AH0 N D AH1 K T"),
    "record": ("R EH1 K ER0 D", "R IH0 K AO1 R D"),
    "present": ("P R EH1 Z AH0 N T", "P R IH0 Z EH1 N T"),
    "object": ("AA1 B JH EH0 K T", "AH0 B JH EH1 K T"),
    "permit": ("P ER1 M IH0 T", "P ER0 M IH1 T"),
    "produce": ("P R OW1 D UW0 S", "P R AH0 D UW1 S"),
    "project": ("P R AA1 JH EH0 K T", "P R AH0 JH EH1 K T"),
    "increase": ("IH1 N K R IY0 S", "IH0 N K R IY1 S"),
}


def _g2p_symbols(text):
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    out = []
    for ph in _g2p(text):
        if ph == " ":
            continue
        if ph in PUNCT_PAUSE:
            if out and out[-1] != "sp":
                out.append("sp")
            continue
        if ph in ARPA:
            out.append(ph)
    return out


def text_to_symbols(text):
    """Normalised text -> list of ARPAbet symbols; punctuation becomes a short pause 'sp'.

    Homographs (e.g. noun vs verb 'conduct') are resolved with an NLTK POS tag, which is what the
    assignment's test sentence E2 probes.
    """
    import nltk
    norm = normalize(text, keep_punct=True)
    tokens = re.findall(r"[A-Za-z']+|[,.;:?!]", norm)
    tags = nltk.pos_tag(tokens)
    out, chunk = [], []

    def flush():
        if chunk:
            out.extend(_g2p_symbols(" ".join(chunk)))
            chunk.clear()

    for tok, tag in tags:
        if tok.lower() in HOMOGRAPHS:
            flush()
            noun, verb = HOMOGRAPHS[tok.lower()]
            out.extend((verb if tag.startswith("VB") else noun).split())
        else:
            chunk.append(tok)
    flush()
    return out


def word_pron(word):
    return [p for p in text_to_symbols(word) if p != "sp"]
