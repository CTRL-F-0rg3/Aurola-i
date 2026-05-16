from wordfreq import top_n_list
import re
import unicodedata

# Słownik PL - możesz rozszerzyć
PL_WORDS = [
    "dom", "kot", "pies", "programowanie", "gra", "system",
    "siec", "komputer", "klucz", "dane", "okno", "plik",
    "funkcja", "zmienna", "petla", "warunek", "klasa", "obiekt"
]

def normalize_token(word):
    """
    Czyści token do bezpiecznej etykiety ASM:
    - usuwa akcenty (café -> cafe)
    - zamienia spacje/myślniki na _
    - usuwa niedozwolone znaki (apostrofy, kropki itp.)
    - dodaje prefix TOK_ jeśli zaczyna się od cyfry
    - zamienia na UPPERCASE
    """
    # Normalizacja Unicode (usuwa akcenty)
    word = unicodedata.normalize("NFD", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")

    # Zamiana separatorów na podkreślnik
    word = word.replace(" ", "_").replace("-", "_").replace(".", "_")

    # Usuń wszystkie znaki poza literami, cyframi i podkreślnikiem
    word = re.sub(r"[^A-Za-z0-9_]", "", word)

    # Nie może być pusty po czyszczeniu
    if not word:
        return None

    # Uppercase
    word = word.upper()

    # Etykieta nie może zaczynać się od cyfry -> prefix TOK_
    if word[0].isdigit():
        word = "TOK_" + word

    return word

def get_english_words(limit=50000):
    return top_n_list("en", limit)

def get_polish_words():
    return PL_WORDS

def build_vocab():
    en = get_english_words(50000)
    pl = get_polish_words()

    # Połącz i posortuj
    all_words = sorted(set(en + pl))

    vocab = {}
    idx = 1  # 0 zarezerwowane dla SPACE

    seen_tokens = set()  # wykrywa kolizje po normalizacji

    for word in all_words:
        token = normalize_token(word)

        if token is None:
            print(f"[SKIP] pusty token dla: '{word}'")
            continue

        if token in seen_tokens:
            print(f"[KOLIZJA] '{word}' -> '{token}' już istnieje, pomijam")
            continue

        seen_tokens.add(token)
        vocab[token] = idx
        idx += 1

    return vocab

def generate_asm(vocab, filename="vocab.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED VOCAB TABLE\n")
        f.write("; token 0 = spacja\n")
        f.write("SPACE EQU 0\n\n")

        for token, idx in vocab.items():
            f.write(f"{token} EQU {idx}\n")

    print(f"Generated {len(vocab)} entries -> {filename}")

if __name__ == "__main__":
    vocab = build_vocab()
    generate_asm(vocab)