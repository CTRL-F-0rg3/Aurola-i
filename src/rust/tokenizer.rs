// src/tokenizer.rs
// Tokenizer wczytuje vocab.asm, vocab_pos.asm, vocab_lemma.asm
// i zamienia tekst na Vec<Token>

use std::collections::HashMap;
use std::fs;

// ========================
// STRUKTURY
// ========================

#[derive(Debug, Clone, PartialEq)]
pub enum PosTag {
    Noun,        // N  - rzeczownik
    Verb,        // V  - czasownik
    Adjective,   // ADJ
    Adverb,      // ADV
    Pronoun,     // PRN
    Preposition, // PRP
    Conjunction, // CNJ
    Numeral,     // NUM
    Interjection,// INT
    ProperNoun,  // NP
    Determiner,  // DET
    Adposition,  // ADP
    Particle,    // PRT
    Symbol,      // SYM
    Punctuation, // PCT
    Unknown,     // UNK
}

impl PosTag {
    pub fn from_id(id: u32) -> Self {
        match id {
            1  => PosTag::Noun,
            2  => PosTag::Verb,
            3  => PosTag::Adjective,
            4  => PosTag::Adverb,
            5  => PosTag::Pronoun,
            6  => PosTag::Preposition,
            7  => PosTag::Conjunction,
            8  => PosTag::Numeral,
            9  => PosTag::Interjection,
            10 => PosTag::ProperNoun,
            11 => PosTag::Determiner,
            12 => PosTag::Adposition,
            13 => PosTag::Particle,
            14 => PosTag::Symbol,
            15 => PosTag::Punctuation,
            _  => PosTag::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            PosTag::Noun         => "N",
            PosTag::Verb         => "V",
            PosTag::Adjective    => "ADJ",
            PosTag::Adverb       => "ADV",
            PosTag::Pronoun      => "PRN",
            PosTag::Preposition  => "PRP",
            PosTag::Conjunction  => "CNJ",
            PosTag::Numeral      => "NUM",
            PosTag::Interjection => "INT",
            PosTag::ProperNoun   => "NP",
            PosTag::Determiner   => "DET",
            PosTag::Adposition   => "ADP",
            PosTag::Particle     => "PRT",
            PosTag::Symbol       => "SYM",
            PosTag::Punctuation  => "PCT",
            PosTag::Unknown      => "UNK",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub word:     String,   // oryginalne słowo z tekstu
    pub id:       u32,      // token ID z vocab.asm
    pub pos:      PosTag,   // kategoria gramatyczna
    pub pos_id:   u32,      // numeryczne ID kategorii
    pub lemma_id: u32,      // ID formy podstawowej
}

impl Token {
    pub fn unknown(word: &str) -> Self {
        Token {
            word:     word.to_string(),
            id:       0,
            pos:      PosTag::Unknown,
            pos_id:   0,
            lemma_id: 0,
        }
    }
}

// ========================
// WCZYTYWANIE PLIKÓW ASM
// ========================

/// Parsuje plik .asm z liniami: ETYKIETA EQU WARTOŚĆ
/// Zwraca HashMap<etykieta, wartość>
fn parse_asm_file(path: &str) -> HashMap<String, u32> {
    let mut map = HashMap::new();

    let content = match fs::read_to_string(path) {
        Ok(c)  => c,
        Err(e) => {
            eprintln!("[WARN] Nie można wczytać {}: {}", path, e);
            return map;
        }
    };

    for line in content.lines() {
        let line = line.trim();

        // Pomijaj komentarze i puste linie
        if line.starts_with(';') || line.is_empty() {
            continue;
        }

        // Format: ETYKIETA EQU WARTOŚĆ  ; opcjonalny komentarz
        let line = match line.split(';').next() {
            Some(l) => l.trim(),
            None    => continue,
        };

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[1] == "EQU" {
            if let Ok(val) = parts[2].parse::<u32>() {
                map.insert(parts[0].to_string(), val);
            }
        }
    }

    map
}

// ========================
// VOCAB TABLES
// ========================

pub struct VocabTables {
    /// słowo -> token_id
    pub vocab:    HashMap<String, u32>,
    /// TOKEN_POS -> pos_id
    pub pos_map:  HashMap<String, u32>,
    /// TOKEN_LEMMA -> lemma_token_id
    pub lemma_map: HashMap<String, u32>,
}

impl VocabTables {
    pub fn load(
        vocab_path: &str,
        pos_path:   &str,
        lemma_path: &str,
    ) -> Self {
        println!("[VOCAB] Wczytywanie {}...", vocab_path);
        let vocab = parse_asm_file(vocab_path);
        println!("[VOCAB] Wczytano {} tokenów", vocab.len());

        println!("[POS]   Wczytywanie {}...", pos_path);
        let pos_map = parse_asm_file(pos_path);
        println!("[POS]   Wczytano {} wpisów", pos_map.len());

        println!("[LEMMA] Wczytywanie {}...", lemma_path);
        let lemma_map = parse_asm_file(lemma_path);
        println!("[LEMMA] Wczytano {} wpisów", lemma_map.len());

        VocabTables { vocab, pos_map, lemma_map }
    }

    /// Szuka token_id dla słowa
    pub fn get_id(&self, word: &str) -> Option<u32> {
        let key = normalize(word);
        self.vocab.get(&key).copied()
    }

    /// Szuka pos_id dla tokenu
    pub fn get_pos_id(&self, token_key: &str) -> u32 {
        let pos_key = format!("{}_POS", token_key);
        self.pos_map.get(&pos_key).copied().unwrap_or(0)
    }

    /// Szuka lemma_id dla tokenu
    pub fn get_lemma_id(&self, token_key: &str) -> u32 {
        let lemma_key = format!("{}_LEMMA", token_key);
        self.lemma_map.get(&lemma_key).copied().unwrap_or(0)
    }
}

// ========================
// NORMALIZACJA
// ========================

/// Normalizuje słowo do formatu etykiety ASM (tak samo jak Python)
pub fn normalize(word: &str) -> String {
    let upper = word.to_uppercase();
    let cleaned: String = upper
        .chars()
        .map(|c| match c {
            ' ' | '-' | '.' => '_',
            '\'' => '\0', // apostrofy usuwamy
            c if c.is_alphanumeric() || c == '_' => c,
            _ => '\0',
        })
        .filter(|&c| c != '\0')
        .collect();

    // Prefix TOK_ jeśli zaczyna się od cyfry
    if cleaned.starts_with(|c: char| c.is_ascii_digit()) {
        format!("TOK_{}", cleaned)
    } else {
        cleaned
    }
}

// ========================
// TOKENIZER
// ========================

pub struct Tokenizer {
    tables: VocabTables,
}

impl Tokenizer {
    pub fn new(tables: VocabTables) -> Self {
        Tokenizer { tables }
    }

    /// Tokenizuje tekst -> Vec<Token>
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let words = self.split_words(text);
        let mut tokens = Vec::with_capacity(words.len());

        for word in words {
            let token = self.process_word(&word);
            tokens.push(token);
        }

        tokens
    }

    fn process_word(&self, word: &str) -> Token {
        let key = normalize(word);

        let id = self.tables.get_id(word).unwrap_or(0);

        if id == 0 {
            // Słowo nieznane
            return Token::unknown(word);
        }

        let pos_id   = self.tables.get_pos_id(&key);
        let lemma_id = self.tables.get_lemma_id(&key);
        let pos      = PosTag::from_id(pos_id);

        Token {
            word: word.to_string(),
            id,
            pos,
            pos_id,
            lemma_id,
        }
    }

    /// Prosta segmentacja tekstu na słowa
    /// Obsługuje interpunkcję jako osobne tokeny
    fn split_words(&self, text: &str) -> Vec<String> {
        let mut words  = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                current.push(ch);
            } else if ch.is_whitespace() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
            } else {
                // Interpunkcja jako osobny token
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                let punct = ch.to_string();
                words.push(punct);
            }
        }

        if !current.is_empty() {
            words.push(current);
        }

        words
    }
}

// ========================
// MAIN (przykład użycia)
// ========================

fn main() {
    // Ścieżki do plików ASM (wszystko w src/)
    let tables = VocabTables::load(
        "src/vocab.asm",
        "src/vocab_pos.asm",
        "src/vocab_lemma.asm",
    );

    let tokenizer = Tokenizer::new(tables);

    let test_sentences = vec![
        "the quick brown fox jumps over the lazy dog",
        "I am running a program",
        "system komputer dane",
    ];

    for sentence in test_sentences {
        println!("\n--- Input: \"{}\" ---", sentence);
        let tokens = tokenizer.tokenize(sentence);

        for t in &tokens {
            println!(
                "  {:20} id={:5}  pos={:4}  lemma_id={:5}  [{}]",
                t.word,
                t.id,
                t.pos.as_str(),
                t.lemma_id,
                if t.id == 0 { "UNKNOWN" } else { "OK" }
            );
        }
    }
}
