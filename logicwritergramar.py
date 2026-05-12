import spacy
import re
from wordfreq import top_n_list

# Załaduj model angielski (python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# ========================
# KONFIGURACJA
# ========================

PL_WORDS = [
    "dom", "kot", "pies", "programowanie", "gra", "system",
    "siec", "komputer", "klucz", "dane", "okno", "plik",
    "funkcja", "zmienna", "petla", "warunek", "klasa", "obiekt"
]

# Mapowanie POS spaCy -> krótki tag
POS_MAP = {
    "NOUN":  "N",   # rzeczownik
    "VERB":  "V",   # czasownik
    "ADJ":   "ADJ", # przymiotnik
    "ADV":   "ADV", # przysłówek
    "PRON":  "PRN", # zaimek
    "PREP":  "PRP", # przyimek
    "CONJ":  "CNJ", # spójnik
    "NUM":   "NUM", # liczebnik
    "INTJ":  "INT", # wykrzyknik
    "PROPN": "NP",  # nazwa własna
    "DET":   "DET", # determinator (the, a)
    "ADP":   "ADP", # przyimek (spaCy)
    "CCONJ": "CNJ",
    "SCONJ": "CNJ",
    "PART":  "PRT", # partykuła
    "SYM":   "SYM", # symbol
    "PUNCT": "PCT", # interpunkcja
    "X":     "UNK", # nieznany
}

# Numeryczne ID dla kategorii POS (do ASM)
POS_ID = {
    "N":   1,
    "V":   2,
    "ADJ": 3,
    "ADV": 4,
    "PRN": 5,
    "PRP": 6,
    "CNJ": 7,
    "NUM": 8,
    "INT": 9,
    "NP":  10,
    "DET": 11,
    "ADP": 12,
    "PRT": 13,
    "SYM": 14,
    "PCT": 15,
    "UNK": 0,
}

# ========================
# NORMALIZACJA (ta sama co w generatorze vocab)
# ========================

import unicodedata

def normalize_token(word):
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
# ANALIZA SŁÓW
# ========================

def analyze_words(word_list, batch_size=1000):
    """
    Analizuje listę słów przez spaCy w batchach.
    Zwraca dict: token_str -> {pos, pos_id, lemma}
    """
    results = {}
    total = len(word_list)

    for i in range(0, total, batch_size):
        batch = word_list[i:i+batch_size]
        # Łączymy słowa spacją żeby spaCy mogło je parsować
        docs = list(nlp.pipe(batch))

        for word, doc in zip(batch, docs):
            token_key = normalize_token(word)
            if token_key is None:
                continue

            if len(doc) == 0:
                pos_tag = "UNK"
            else:
                # Bierzemy POS pierwszego tokenu (słowo pojedyncze)
                spacy_pos = doc[0].pos_
                pos_tag = POS_MAP.get(spacy_pos, "UNK")

            lemma = doc[0].lemma_.upper() if len(doc) > 0 else word.upper()
            lemma_key = normalize_token(lemma) or token_key

            results[token_key] = {
                "pos":     pos_tag,
                "pos_id":  POS_ID.get(pos_tag, 0),
                "lemma":   lemma_key,
                "original": word.upper()
            }

        print(f"  Przeanalizowano {min(i+batch_size, total)}/{total} słów...")

    return results

# ========================
# GENERATORY PLIKÓW
# ========================

def generate_pos_asm(analysis, filename="vocab_pos.asm"):
    """
    Generuje tabelę POS w ASM.
    Każdy token ma stałą: TOKEN_POS_<NAZWA> EQU <pos_id>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED POS TABLE\n")
        f.write("; Format: <TOKEN>_POS EQU <pos_id>\n")
        f.write("; POS IDs: 0=UNK 1=N 2=V 3=ADJ 4=ADV 5=PRN\n")
        f.write(";          6=PRP 7=CNJ 8=NUM 9=INT 10=NP\n")
        f.write(";          11=DET 12=ADP 13=PRT 14=SYM 15=PCT\n\n")

        for token, info in sorted(analysis.items()):
            f.write(f"{token}_POS EQU {info['pos_id']}  ; {info['pos']}\n")

    print(f"[ASM] Zapisano {len(analysis)} wpisów -> {filename}")

def generate_lemma_asm(analysis, vocab, filename="vocab_lemma.asm"):
    """
    Generuje tabelę lemmatów: TOKEN_LEMMA EQU <token_id_lemmatu>
    Żeby AI mogło redukować formy do rdzenia.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED LEMMA TABLE\n")
        f.write("; Format: <TOKEN>_LEMMA EQU <token_id_lemmatu>\n\n")

        for token, info in sorted(analysis.items()):
            lemma = info["lemma"]
            lemma_id = vocab.get(lemma, 0)  # 0 jeśli lemma nie ma w vocab
            f.write(f"{token}_LEMMA EQU {lemma_id}  ; {lemma}\n")

    print(f"[ASM] Zapisano {len(analysis)} wpisów -> {filename}")

def generate_forth_dict(analysis, vocab, filename="vocab.forth"):
    """
    Generuje słownik FORTH.
    Każde słowo to definicja zwracająca [token_id, pos_id, lemma_id]
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("( AUTO GENERATED FORTH VOCAB DICTIONARY )\n")
        f.write("( Użycie: <SŁOWO> -> token_id pos_id lemma_id )\n\n")

        for token, info in sorted(analysis.items()):
            token_id  = vocab.get(token, 0)
            pos_id    = info["pos_id"]
            lemma_id  = vocab.get(info["lemma"], 0)

            # Definicja FORTH: wywołanie słowa zostawia 3 wartości na stosie
            f.write(f": {token} ( -- token_id pos_id lemma_id )\n")
            f.write(f"  {token_id} {pos_id} {lemma_id} ;\n\n")

    print(f"[FORTH] Zapisano {len(analysis)} wpisów -> {filename}")

def generate_logic_map(analysis, vocab, filename="logic_map.asm"):
    """
    Mapa logiczna: grupy tokenów według kategorii POS.
    Ułatwia AI sprawdzenie 'czy token X jest czasownikiem'.
    Format: bloki etykiet dla każdej kategorii.
    """
    # Grupuj tokeny po POS
    groups = {}
    for token, info in analysis.items():
        pos = info["pos"]
        if pos not in groups:
            groups[pos] = []
        groups[pos].append((token, vocab.get(token, 0)))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED LOGIC MAP\n")
        f.write("; Grupy tokenów według kategorii gramatycznej\n\n")

        for pos_tag, tokens in sorted(groups.items()):
            pos_id = POS_ID.get(pos_tag, 0)
            f.write(f"; ===== {pos_tag} (POS_ID={pos_id}) =====\n")
            f.write(f"POS_{pos_tag}_START EQU {tokens[0][1]}\n")
            f.write(f"POS_{pos_tag}_COUNT EQU {len(tokens)}\n")

            for token, tok_id in sorted(tokens, key=lambda x: x[1]):
                f.write(f"  ; {token} = {tok_id}\n")

            f.write("\n")

    print(f"[ASM] Mapa logiczna zapisana -> {filename}")

# ========================
# GŁÓWNY PIPELINE
# ========================

def load_vocab_ids(filename="vocab.asm"):
    """
    Wczytuje istniejący vocab.asm (z poprzedniego generatora)
    żeby mieć spójne ID tokenów.
    """
    vocab = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(";") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "EQU":
                try:
                    vocab[parts[0]] = int(parts[2])
                except ValueError:
                    pass
    print(f"[VOCAB] Wczytano {len(vocab)} tokenów z {filename}")
    return vocab

if __name__ == "__main__":
    print("=== KROK 1: Wczytywanie vocab ===")
    vocab = load_vocab_ids("vocab.asm")

    print("=== KROK 2: Pobieranie listy słów ===")
    en_words = top_n_list("en", 50000)
    all_words = list(set(en_words + PL_WORDS))
    print(f"  Łącznie słów do analizy: {len(all_words)}")

    print("=== KROK 3: Analiza POS przez spaCy ===")
    analysis = analyze_words(all_words, batch_size=500)

    print("=== KROK 4: Generowanie plików ===")
    generate_pos_asm(analysis, "vocab_pos.asm")
    generate_lemma_asm(analysis, vocab, "vocab_lemma.asm")
    generate_forth_dict(analysis, vocab, "vocab.forth")
    generate_logic_map(analysis, vocab, "logic_map.asm")

    print("\n=== GOTOWE ===")
    print("Wygenerowane pliki:")
    print("  vocab_pos.asm   - kategorie gramatyczne (POS)")
    print("  vocab_lemma.asm - lemmaty (formy podstawowe)")
    print("  vocab.forth     - słownik FORTH")
    print("  logic_map.asm   - mapa logiczna grup")