# src/grammar_gen.py
# Generator reguł gramatycznych -> pliki ASM
# Uruchamiany przez GUI lub ręcznie

import re

# ========================
# REGUŁY GRAMATYCZNE
# ========================

# Format: (nazwa_reguły, lewa_strona, prawa_strona_jako_lista_POS)
# POS: N=rzeczownik V=czasownik ADJ=przymiotnik ADV=przysłówek
#      DET=determinator PRN=zaimek CNJ=spójnik PRP=przyimek NP=nazwa_własna

CFG_RULES = [
    # Zdania
    ("S_BASIC",     "S",   ["NP", "VP"]),
    ("S_QUESTION",  "S",   ["AUX", "NP", "VP"]),
    ("S_PASSIVE",   "S",   ["NP", "AUX", "V_PAST", "PRP", "NP"]),
    ("S_COMPOUND",  "S",   ["S", "CNJ", "S"]),

    # Frazy nominalne
    ("NP_FULL",     "NP",  ["DET", "ADJ", "N"]),
    ("NP_SIMPLE",   "NP",  ["DET", "N"]),
    ("NP_BARE",     "NP",  ["N"]),
    ("NP_PRONOUN",  "NP",  ["PRN"]),
    ("NP_PROPER",   "NP",  ["NP_NAME"]),
    ("NP_PREP",     "NP",  ["NP", "PP"]),
    ("NP_MULTI_ADJ","NP",  ["DET", "ADJ", "ADJ", "N"]),

    # Frazy werbalne
    ("VP_TRANS",    "VP",  ["V", "NP"]),
    ("VP_INTRANS",  "VP",  ["V"]),
    ("VP_ADV",      "VP",  ["V", "ADV"]),
    ("VP_PREP",     "VP",  ["V", "PP"]),
    ("VP_NP_PP",    "VP",  ["V", "NP", "PP"]),
    ("VP_MODAL",    "VP",  ["MOD", "V"]),
    ("VP_MODAL_NP", "VP",  ["MOD", "V", "NP"]),

    # Frazy przyimkowe
    ("PP_BASIC",    "PP",  ["PRP", "NP"]),

    # Frazy przymiotnikowe
    ("ADJP_BASIC",  "ADJP",["ADJ"]),
    ("ADJP_ADV",    "ADJP",["ADV", "ADJ"]),

    # Frazy przysłówkowe
    ("ADVP_BASIC",  "ADVP",["ADV"]),
    ("ADVP_PREP",   "ADVP",["PRP", "NP"]),
]

# Końcówki odmian angielskich czasowników
VERB_ENDINGS = [
    ("3SG_S",     "_S",     "He/She/It walks"),
    ("PROG_ING",  "_ING",   "Running, jumping"),
    ("PAST_ED",   "_ED",    "Walked, jumped"),
    ("PAST_IRR",  "_IRREG", "Ran, went, saw"),
    ("INF_TO",    "TO_",    "To run, to go"),
]

# Końcówki liczby mnogiej rzeczowników
NOUN_ENDINGS = [
    ("PL_S",      "_S",     "cats, dogs"),
    ("PL_ES",     "_ES",    "boxes, matches"),
    ("PL_IES",    "_IES",   "cities, babies"),
    ("PL_IRR",    "_IRR",   "children, mice"),
    ("POSS_S",    "_APOS_S","cat's, dog's"),
]

# Stopniowanie przymiotników
ADJ_ENDINGS = [
    ("COMP_ER",   "_ER",    "bigger, faster"),
    ("COMP_MORE", "MORE_",  "more beautiful"),
    ("SUPER_EST", "_EST",   "biggest, fastest"),
    ("SUPER_MOST","MOST_",  "most beautiful"),
]

# Słowa kluczowe gramatyki
GRAMMAR_KEYWORDS = {
    # Determinatory
    "DETERMINERS": [
        "THE", "A", "AN", "THIS", "THAT", "THESE", "THOSE",
        "MY", "YOUR", "HIS", "HER", "ITS", "OUR", "THEIR",
        "SOME", "ANY", "NO", "EACH", "EVERY", "EITHER",
    ],
    # Zaimki
    "PRONOUNS": [
        "I", "YOU", "HE", "SHE", "IT", "WE", "THEY",
        "ME", "HIM", "HER_PRN", "US", "THEM",
        "MYSELF", "YOURSELF", "HIMSELF", "HERSELF",
        "WHO", "WHAT", "WHICH", "THAT_PRN",
    ],
    # Spójniki
    "CONJUNCTIONS": [
        "AND", "OR", "BUT", "NOR", "FOR", "YET", "SO",
        "BECAUSE", "ALTHOUGH", "WHILE", "WHEN", "IF",
        "UNLESS", "UNTIL", "SINCE", "AFTER", "BEFORE",
    ],
    # Przyimki
    "PREPOSITIONS": [
        "IN", "ON", "AT", "TO", "FOR", "WITH", "BY",
        "FROM", "OF", "ABOUT", "THROUGH", "BETWEEN",
        "UNDER", "OVER", "AFTER", "BEFORE", "DURING",
        "INTO", "ONTO", "WITHIN", "WITHOUT", "AGAINST",
    ],
    # Czasowniki modalne
    "MODALS": [
        "CAN", "COULD", "WILL", "WOULD", "SHALL", "SHOULD",
        "MAY", "MIGHT", "MUST", "OUGHT", "NEED_MOD", "DARE",
    ],
    # Czasowniki posiłkowe
    "AUXILIARIES": [
        "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
        "HAS", "HAVE", "HAD", "DO", "DOES", "DID",
    ],
    # Przysłówki miejsca/czasu
    "ADVERBS_PLACE": [
        "HERE", "THERE", "EVERYWHERE", "NOWHERE", "SOMEWHERE",
        "INSIDE", "OUTSIDE", "NEARBY", "AWAY",
    ],
    "ADVERBS_TIME": [
        "NOW", "THEN", "TODAY", "YESTERDAY", "TOMORROW",
        "ALWAYS", "NEVER", "OFTEN", "SOMETIMES", "USUALLY",
        "ALREADY", "STILL", "YET_ADV", "SOON", "RECENTLY",
    ],
}

# Numeryczne ID kategorii reguł
RULE_CATEGORY_ID = {
    "S":    1,   # zdanie
    "NP":   2,   # fraza nominalna
    "VP":   3,   # fraza werbalna
    "PP":   4,   # fraza przyimkowa
    "ADJP": 5,   # fraza przymiotnikowa
    "ADVP": 6,   # fraza przysłówkowa
}

POS_TO_ID = {
    "N":      1,
    "V":      2,
    "ADJ":    3,
    "ADV":    4,
    "PRN":    5,
    "PRP":    6,
    "CNJ":    7,
    "NUM":    8,
    "DET":    11,
    "AUX":    16,
    "MOD":    17,
    "NP_NAME":10,
    "V_PAST": 18,
    "S":      19,
}

# ========================
# NORMALIZACJA
# ========================

def normalize(word: str) -> str:
    word = word.upper()
    word = re.sub(r"[^A-Z0-9_]", "_", word)
    if word and word[0].isdigit():
        word = "TOK_" + word
    return word

# ========================
# GENERATORY
# ========================

def generate_cfg_rules(filename="src/grammar_svo.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED CFG GRAMMAR RULES\n")
        f.write("; Format: RULE_<NAZWA>_LHS EQU <category_id>\n")
        f.write(";         RULE_<NAZWA>_LEN EQU <liczba_symboli>\n")
        f.write(";         RULE_<NAZWA>_0   EQU <pos_id_0>\n")
        f.write(";         RULE_<NAZWA>_1   EQU <pos_id_1>  ...\n\n")

        f.write("; Kategorie: 1=S 2=NP 3=VP 4=PP 5=ADJP 6=ADVP\n")
        f.write("; POS IDs:   1=N 2=V 3=ADJ 4=ADV 5=PRN 6=PRP\n")
        f.write(";            7=CNJ 11=DET 16=AUX 17=MOD 18=V_PAST 19=S\n\n")

        for rule_name, lhs, rhs in CFG_RULES:
            lhs_id  = RULE_CATEGORY_ID.get(lhs, 0)
            f.write(f"; --- {rule_name}: {lhs} -> {' '.join(rhs)} ---\n")
            f.write(f"RULE_{rule_name}_LHS EQU {lhs_id}\n")
            f.write(f"RULE_{rule_name}_LEN EQU {len(rhs)}\n")
            for i, sym in enumerate(rhs):
                sym_id = POS_TO_ID.get(sym, 0)
                f.write(f"RULE_{rule_name}_{i}   EQU {sym_id}  ; {sym}\n")
            f.write("\n")

    print(f"[GRAMMAR] {len(CFG_RULES)} reguł CFG -> {filename}")

def generate_endings(filename="src/grammar_endings.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED MORPHOLOGICAL ENDINGS\n\n")

        f.write("; === CZASOWNIKI ===\n")
        for idx, (name, suffix, comment) in enumerate(VERB_ENDINGS, 1):
            safe = normalize(suffix)
            f.write(f"VERB_END_{name} EQU {idx}  ; {comment} (suffix: {suffix})\n")
        f.write("\n")

        f.write("; === RZECZOWNIKI ===\n")
        for idx, (name, suffix, comment) in enumerate(NOUN_ENDINGS, 1):
            f.write(f"NOUN_END_{name} EQU {idx}  ; {comment}\n")
        f.write("\n")

        f.write("; === PRZYMIOTNIKI ===\n")
        for idx, (name, suffix, comment) in enumerate(ADJ_ENDINGS, 1):
            f.write(f"ADJ_END_{name} EQU {idx}  ; {comment}\n")

    print(f"[ENDINGS] Końcówki morfologiczne -> {filename}")

def generate_keywords(filename="src/grammar_keywords.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED GRAMMAR KEYWORDS\n")
        f.write("; Słowa funkcyjne podzielone na kategorie\n\n")

        global_idx = 1
        for category, words in GRAMMAR_KEYWORDS.items():
            f.write(f"; === {category} ===\n")
            f.write(f"KW_{category}_START EQU {global_idx}\n")
            f.write(f"KW_{category}_COUNT EQU {len(words)}\n")
            for word in words:
                safe = normalize(word)
                f.write(f"KW_{safe} EQU {global_idx}  ; {category}\n")
                global_idx += 1
            f.write("\n")

    total = sum(len(w) for w in GRAMMAR_KEYWORDS.values())
    print(f"[KEYWORDS] {total} słów kluczowych -> {filename}")

def generate_forth_grammar(filename="src/grammar.forth"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("( AUTO GENERATED GRAMMAR FORTH WORDS )\n\n")

        f.write("( Sprawdzenie czy POS pasuje do kategorii )\n")
        f.write(": IS-NOUN?    ( pos_id -- bool ) 1 = ;\n")
        f.write(": IS-VERB?    ( pos_id -- bool ) 2 = ;\n")
        f.write(": IS-ADJ?     ( pos_id -- bool ) 3 = ;\n")
        f.write(": IS-ADV?     ( pos_id -- bool ) 4 = ;\n")
        f.write(": IS-PRONOUN? ( pos_id -- bool ) 5 = ;\n")
        f.write(": IS-PREP?    ( pos_id -- bool ) 6 = ;\n")
        f.write(": IS-CONJ?    ( pos_id -- bool ) 7 = ;\n")
        f.write(": IS-DET?     ( pos_id -- bool ) 11 = ;\n")
        f.write(": IS-AUX?     ( pos_id -- bool ) 16 = ;\n")
        f.write(": IS-MODAL?   ( pos_id -- bool ) 17 = ;\n")
        f.write("\n")

        f.write("( Reguły gramatyczne — sprawdź czy sekwencja POS pasuje )\n")
        for rule_name, lhs, rhs in CFG_RULES:
            comment = f"{lhs} -> {' '.join(rhs)}"
            f.write(f"( {rule_name}: {comment} )\n")
            # FORTH: na stosie pos_id każdego tokenu, zwraca bool
            checks = " AND ".join([f"IS-{sym.replace('_NAME','').replace('_PAST','')}?" for sym in rhs
                                    if sym in ["N","V","ADJ","ADV","PRN","PRP","CNJ","DET","AUX","MOD"]])
            if checks:
                f.write(f": MATCH-{rule_name} ( -- ) {checks} ;\n\n")

    print(f"[FORTH] Gramatyka FORTH -> {filename}")

# ========================
# MAIN
# ========================

if __name__ == "__main__":
    print("=== GENERATOR GRAMATYKI AURORA ===\n")

    generate_cfg_rules("src/grammar_svo.asm")
    generate_endings("src/grammar_endings.asm")
    generate_keywords("src/grammar_keywords.asm")
    generate_forth_grammar("src/grammar.forth")

    print("\n=== GOTOWE ===")
    print("  src/grammar_svo.asm      - reguły CFG")
    print("  src/grammar_endings.asm  - końcówki morfologiczne")
    print("  src/grammar_keywords.asm - słowa funkcyjne")
    print("  src/grammar.forth        - słownik FORTH gramatyki")
