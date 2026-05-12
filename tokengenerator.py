from wordfreq import top_n_list
import nltk
from nltk.corpus import words

# opcjonalnie PL (prosty słownik)
PL_WORDS = [
    "dom", "kot", "pies", "programowanie", "gra", "system",
    "sieć", "komputer", "klucz", "dane"
]

def get_english_words(limit=50000):
    return top_n_list("en", limit)

def get_polish_words():
    # nltk nie ma dobrego PL lexiconu, więc mieszamy własne + opcjonalnie rozszerzenia
    return PL_WORDS

def build_vocab():
    en = get_english_words(50000)
    pl = get_polish_words()

    all_words = set(en + pl)

    vocab = {}
    idx = 1

    for w in sorted(all_words):
        vocab[w] = idx
        idx += 1

    return vocab

def generate_asm(vocab, filename="vocab.asm"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("; AUTO GENERATED VOCAB TABLE\n\n")

        for word, idx in vocab.items():
            safe_word = word.replace(" ", "_").replace("-", "_")
            f.write(f"{safe_word.upper()} EQU {idx}\n")

    print(f"Generated {len(vocab)} entries -> {filename}")

if __name__ == "__main__":
    vocab = build_vocab()
    generate_asm(vocab)
