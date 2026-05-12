// build.rs
// Uruchamiany przez cargo przed kompilacją.
// Sprawdza czy pliki ASM istnieją i liczy tokeny.

use std::fs;
use std::path::Path;

fn count_tokens(path: &str) -> usize {
    match fs::read_to_string(path) {
        Ok(content) => content
            .lines()
            .filter(|l| {
                let l = l.trim();
                !l.starts_with(';') && !l.is_empty() && l.contains("EQU")
            })
            .count(),
        Err(_) => 0,
    }
}

fn main() {
    let asm_files = [
        "src/vocab.asm",
        "src/vocab_pos.asm",
        "src/vocab_lemma.asm",
        "src/logic_map.asm",
    ];

    for file in &asm_files {
        if Path::new(file).exists() {
            let count = count_tokens(file);
            println!("cargo:warning={} -> {} wpisów", file, count);
        } else {
            // Nie przerywaj buildu — pliki ASM mogą być generowane później
            println!("cargo:warning=[BRAK] {} - uruchom generator Python", file);
        }

        // Przebuduj jeśli plik ASM się zmienił
        println!("cargo:rerun-if-changed={}", file);
    }

    // Przebuduj jeśli sam build.rs się zmienił
    println!("cargo:rerun-if-changed=build.rs");
}
