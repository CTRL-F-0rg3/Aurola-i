# src/vocab_gen.py
# Generator słownika EN — używa TAG (fine-grained) zamiast POS
# Rozwiązuje problem: 'the','a' tagowane jako PRON zamiast DET

import re
import unicodedata
import spacy
from wordfreq import top_n_list

nlp = spacy.load("en_core_web_sm")

# ========================
# MAPOWANIE TAG -> POS_ID
# Używamy fine-grained TAG bo jest dokładniejszy dla pojedynczych słów
# ========================

TAG_TO_POS_ID = {
    # Rzeczowniki -> 1
    "NN":  1, "NNS": 1, "NNP": 1, "NNPS": 1,

    # Czasowniki -> 2
    "VB":  2, "VBD": 2, "VBG": 2,
    "VBN": 2, "VBP": 2, "VBZ": 2,

    # Przymiotniki -> 3
    "JJ":  3, "JJR": 3, "JJS": 3,

    # Przysłówki -> 4
    "RB":  4, "RBR": 4, "RBS": 4, "WRB": 4,

    # Zaimki -> 5
    "PRP": 5, "PRP$": 5, "WP": 5, "WP$": 5,

    # Przyimki -> 6
    "IN":  6,

    # Spójniki -> 7
    "CC":  7,

    # Liczebniki -> 8
    "CD":  8,

    # Wykrzykniki -> 9
    "UH":  9,

    # Determinatory -> 11  (DT zawsze 11, niezależnie od POS)
    "DT":  11, "PDT": 11, "WDT": 11,

    # Czasowniki posiłkowe -> 16
    "MD":  16,  # can, could, will, would, shall, should, may, might, must

    # Partykuły -> 13
    "RP":  13, "TO": 13,

    # Symbole -> 14
    "SYM": 14, "$": 14,

    # Interpunkcja -> 15
    ".":   15, ",": 15, ":": 15, "``": 15, "''": 15,
    "-LRB-": 15, "-RRB-": 15,

    # Nazwy własne -> 10
    "NNP": 10, "NNPS": 10,

    # Nieznane -> 0
    "XX":  0, "AFX": 0, "GW": 0,
    "ADD": 0, "NFP": 0, "FW": 0,
    "LS":  0, "NIL": 0,
}

# Słowa które zawsze mają konkretny POS niezależnie od taggera
FORCE_POS = {
    # Determinatory
    "the": 11, "a": 11, "an": 11, "this": 11, "that": 11,
    "these": 11, "those": 11, "my": 11, "your": 11, "his": 11,
    "her": 11, "its": 11, "our": 11, "their": 11, "every": 11,
    "each": 11, "some": 11, "any": 11, "no": 11, "both": 11,
    "all": 11, "half": 11, "either": 11, "neither": 11,

    # Posiłkowe
    "is": 16, "are": 16, "was": 16, "were": 16, "be": 16,
    "been": 16, "being": 16, "am": 16,
    "has": 16, "have": 16, "had": 16,
    "do": 16, "does": 16, "did": 16,

    # Modalne
    "can": 16, "could": 16, "will": 16, "would": 16,
    "shall": 16, "should": 16, "may": 16, "might": 16,
    "must": 16, "ought": 16,

    # Zaimki pytające
    "what": 5, "who": 5, "whom": 5, "whose": 5, "which": 5,

    # Przysłówki pytające
    "how": 4, "when": 4, "where": 4, "why": 4,

    # Spójniki
    "and": 7, "or": 7, "but": 7, "nor": 7, "for": 6,
    "yet": 7, "so": 7, "because": 7, "although": 7,
    "while": 7, "if": 7, "unless": 7, "until": 7,
    "since": 7, "after": 7, "before": 7, "though": 7,

    # Przyimki
    "in": 6, "on": 6, "at": 6, "to": 6, "with": 6,
    "by": 6, "from": 6, "of": 6, "about": 6, "through": 6,
    "between": 6, "under": 6, "over": 6, "into": 6,
    "onto": 6, "within": 6, "without": 6, "against": 6,
    "along": 6, "around": 6, "beside": 6, "beyond": 6,

    # Zaimki osobowe
    "i": 5, "you": 5, "he": 5, "she": 5, "it": 5,
    "we": 5, "they": 5, "me": 5, "him": 5, "us": 5, "them": 5,
    "myself": 5, "yourself": 5, "himself": 5, "herself": 5,
    "itself": 5, "ourselves": 5, "themselves": 5,
}

# ========================
# NORMALIZACJA
# ========================

def normalize(word: str) -> str:
    word = unicodedata.normalize("NFD", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")
    word = word.replace(" ", "_").replace("-", "_").replace(".", "_")
    word = re.sub(r"[^A-Za-z0-9_]", "", word)
    if not word:
        return None
    word = word.upper()
    if word[0].isdigit():
        word = "TOK_" + word
    return word

# ========================
# ANALIZA
# ========================

def get_pos_id(word: str, doc) -> int:
    """Pobiera POS_ID używając fine-grained TAG + FORCE_POS"""
    lower = word.lower()

    # Najpierw sprawdź FORCE_POS
    if lower in FORCE_POS:
        return FORCE_POS[lower]

    # Potem fine-grained TAG
    if len(doc) > 0:
        tag = doc[0].tag_
        if tag in TAG_TO_POS_ID:
            return TAG_TO_POS_ID[tag]

        # Fallback na coarse POS
        coarse = doc[0].pos_
        fallback = {
            "NOUN": 1, "VERB": 2, "ADJ": 3, "ADV": 4,
            "PRON": 5, "ADP": 6, "CCONJ": 7, "SCONJ": 7,
            "NUM": 8, "INTJ": 9, "PROPN": 10, "DET": 11,
            "AUX": 16, "PART": 13, "SYM": 14, "PUNCT": 15,
        }
        return fallback.get(coarse, 0)

    return 0

def analyze_batch(words: list, batch_size: int = 500) -> dict:
    """Analizuje słowa przez spaCy, zwraca {word: {pos_id, lemma}}"""
    results = {}
    total = len(words)

    for i in range(0, total, batch_size):
        batch = words[i:i+batch_size]
        docs  = list(nlp.pipe(batch))

        for word, doc in zip(batch, docs):
            key = normalize(word)
            if key is None:
                continue

            pos_id = get_pos_id(word, doc)
            lemma  = normalize(doc[0].lemma_ if len(doc) > 0 else word) or key

            results[key] = {
                "pos_id": pos_id,
                "lemma":  lemma,
                "original": word,
            }

        print(f"  [{min(i+batch_size, total)}/{total}] przeanalizowano...")

    return results

# ========================
# GENERATORY ASM
# ========================

def generate_vocab(analysis: dict, filename: str = "src/vocab.asm"):
    seen   = set()
    vocab  = {}
    idx    = 1

    for token in sorted(analysis.keys()):
        if token in seen:
            continue
        seen.add(token)
        vocab[token] = idx
        idx += 1

    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED VOCAB TABLE\n")
        f.write("; token 0 = spacja\n")
        f.write("SPACE EQU 0\n\n")
        for token, tid in vocab.items():
            f.write(f"{token} EQU {tid}\n")

    print(f"[VOCAB] {len(vocab)} tokenów -> {filename}")
    return vocab

def generate_pos(analysis: dict, filename: str = "src/vocab_pos.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED POS TABLE\n")
        f.write("; POS IDs: 0=UNK 1=N 2=V 3=ADJ 4=ADV 5=PRN\n")
        f.write(";          6=PRP 7=CNJ 8=NUM 9=INT 10=NP\n")
        f.write(";          11=DET 12=ADP 13=PRT 14=SYM 15=PCT 16=AUX\n\n")

        for token, info in sorted(analysis.items()):
            f.write(f"{token}_POS EQU {info['pos_id']}\n")

    print(f"[POS]   {len(analysis)} wpisów -> {filename}")

def generate_lemma(analysis: dict, vocab: dict, filename: str = "src/vocab_lemma.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED LEMMA TABLE\n\n")

        for token, info in sorted(analysis.items()):
            lemma_id = vocab.get(info["lemma"], 0)
            f.write(f"{token}_LEMMA EQU {lemma_id}\n")

    print(f"[LEMMA] {len(analysis)} wpisów -> {filename}")

def verify_key_words(analysis: dict):
    """Sprawdź czy kluczowe słowa mają poprawne POS"""
    check = {
        "THE": 11, "A": 11, "AN": 11,
        "IS": 16,  "ARE": 16, "WAS": 16,
        "CAN": 16, "WILL": 16, "SHOULD": 16,
        "WHAT": 5, "WHO": 5, "HOW": 4,
        "IN": 6,   "ON": 6,  "AT": 6,
        "AND": 7,  "OR": 7,  "BUT": 7,
        "RUN": 2,  "MAKE": 2, "GO": 2,
        "BIG": 3,  "FAST": 3, "GOOD": 3,
    }
    print("\n=== WERYFIKACJA KLUCZOWYCH SŁÓW ===")
    errors = 0
    for word, expected in check.items():
        actual = analysis.get(word, {}).get("pos_id", -1)
        status = "OK" if actual == expected else f"BŁĄD (oczekiwano {expected}, got {actual})"
        if actual != expected:
            errors += 1
        print(f"  {word:10} POS_ID={actual:3}  {status}")
    print(f"Błędów: {errors}/{len(check)}\n")

# ========================
# MAIN
# ========================

if __name__ == "__main__":
    print("=== GENERATOR VOCAB (fixed POS) ===\n")

    print("Pobieranie słów EN...")
    words = top_n_list("en", 50000)
    print(f"Pobrano {len(words)} słów\n")

    print("Analiza POS przez spaCy (TAG fine-grained)...")
    analysis = analyze_batch(words)

    print("\nGenerowanie plików ASM...")
    vocab = generate_vocab(analysis)
    generate_pos(analysis)
    generate_lemma(analysis, vocab)

    verify_key_words(analysis)

    print("=== GOTOWE ===")
    print("  src/vocab.asm")
    print("  src/vocab_pos.asm")
    print("  src/vocab_lemma.asm")
