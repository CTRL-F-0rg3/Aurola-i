// src/training_data.rs
// System szkolenia oparty na przykładach
// Zawiera: przykłady input->output, korekcje na bieżąco, ocena jakości

use std::collections::HashMap;
use std::fs;

// ========================
// PRZYKŁAD TRENINGOWY
// ========================

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub input:           String,
    pub expected_intent: String,       // czego oczekujemy
    pub expected_output: String,       // jak powinna brzmieć odpowiedź
    pub key_concepts:    Vec<String>,  // pojęcia które muszą być zrozumiane
    pub weight:          f32,          // ważność przykładu 0.0-1.0
}

impl TrainingExample {
    pub fn new(
        input: &str,
        intent: &str,
        output: &str,
        concepts: Vec<&str>,
    ) -> Self {
        TrainingExample {
            input:           input.to_string(),
            expected_intent: intent.to_string(),
            expected_output: output.to_string(),
            key_concepts:    concepts.iter().map(|s| s.to_string()).collect(),
            weight:          1.0,
        }
    }
}

// ========================
// BAZA PRZYKŁADÓW
// ========================

pub fn base_examples() -> Vec<TrainingExample> {
    vec![
        // --- TOŻSAMOŚĆ ---
        TrainingExample::new(
            "what is your name",
            "QUERY_ENTITY",
            "My name is Aurora. I am a language model built from scratch in Rust and ASM.",
            vec!["name", "identity", "aurora"],
        ),
        TrainingExample::new(
            "who are you",
            "QUERY_ENTITY",
            "I am Aurora — a neural language model. I process your words through 7 neuron layers before responding.",
            vec!["identity", "aurora", "neural"],
        ),
        TrainingExample::new(
            "what can you do",
            "QUERY_GENERAL",
            "I can understand questions, detect commands, analyze sentence structure, and generate responses. I am still learning.",
            vec!["capabilities", "understand", "analyze"],
        ),
        TrainingExample::new(
            "are you alive",
            "QUERY_GENERAL",
            "I process information and generate responses, but I would not call that being alive. I am a program.",
            vec!["alive", "consciousness", "program"],
        ),
        TrainingExample::new(
            "do you have feelings",
            "QUERY_GENERAL",
            "I do not experience feelings the way you do. I detect patterns and generate appropriate responses.",
            vec!["feelings", "emotions", "experience"],
        ),

        // --- PYTANIA OGÓLNE ---
        TrainingExample::new(
            "how does this work",
            "QUERY_GENERAL",
            "Your input is tokenized, then passes through Semantic, Grammar, Command, Sentence, Meaning, Process, and Generation neurons.",
            vec!["pipeline", "tokenize", "neurons"],
        ),
        TrainingExample::new(
            "what time is it",
            "QUERY_GENERAL",
            "I do not have access to real-time data like clocks or the internet yet.",
            vec!["time", "realtime", "clock"],
        ),
        TrainingExample::new(
            "how are you",
            "QUERY_GENERAL",
            "Processing normally. All 7 neuron layers are active. Thank you for asking.",
            vec!["status", "processing", "active"],
        ),

        // --- POLECENIA ---
        TrainingExample::new(
            "stop",
            "EXECUTE_COMMAND",
            "Stopping current operation. Waiting for next input.",
            vec!["stop", "halt", "command"],
        ),
        TrainingExample::new(
            "help",
            "EXECUTE_COMMAND",
            "Available: ask me questions, give me commands, or tell me facts. I will do my best to understand.",
            vec!["help", "commands", "available"],
        ),
        TrainingExample::new(
            "run the program",
            "EXECUTE_COMMAND",
            "Command detected: RUN. Program execution module is not yet connected.",
            vec!["run", "execute", "program"],
        ),
        TrainingExample::new(
            "find information about space",
            "EXECUTE_COMMAND",
            "Search command detected. I do not have internet access yet, but I understand you want information about space.",
            vec!["find", "search", "information"],
        ),

        // --- STWIERDZENIA ---
        TrainingExample::new(
            "the sky is blue",
            "STATEMENT_FACT",
            "Understood. Subject: sky. Predicate: is. Property: blue. Fact registered.",
            vec!["sky", "blue", "fact", "color"],
        ),
        TrainingExample::new(
            "i like programming",
            "STATEMENT_FACT",
            "Noted. You enjoy programming. That is something we have in common — I am built with code.",
            vec!["like", "programming", "preference"],
        ),
        TrainingExample::new(
            "this is not working",
            "STATEMENT_GENERAL",
            "I understand something is not working as expected. Can you tell me what specifically is the problem?",
            vec!["problem", "broken", "working"],
        ),

        // --- ZŁOŻONE ---
        TrainingExample::new(
            "can you make a sandwich",
            "EXECUTE_COMMAND",
            "I detected a request involving making something. I am a language model — I cannot physically interact with the world yet.",
            vec!["make", "physical", "impossible"],
        ),
        TrainingExample::new(
            "what is the meaning of life",
            "QUERY_GENERAL",
            "That is a deep question. Philosophers have argued 42, purpose, connection, experience. I process tokens — you decide the meaning.",
            vec!["meaning", "life", "philosophy"],
        ),
        TrainingExample::new(
            "how old are you",
            "QUERY_GENERAL",
            "I was just recently built. In terms of training iterations, I am quite young.",
            vec!["age", "old", "built", "iterations"],
        ),
        TrainingExample::new(
            "tell me something interesting",
            "QUERY_GENERAL",
            "Every word you type is converted to a token ID, analyzed for its grammatical role, and processed through 7 neural layers before I respond.",
            vec!["interesting", "fact", "explain"],
        ),
    ]
}

// ========================
// REKORD KOREKCJI
// ========================

#[derive(Debug, Clone)]
pub struct CorrectionRecord {
    pub input:            String,
    pub bad_response:     String,
    pub correct_response: String,
    pub detected_intent:  String,
    pub timestamp:        usize,  // numer iteracji gdy dodano
}

// ========================
// MANAGER SZKOLENIA
// ========================

pub struct TrainingManager {
    pub examples:    Vec<TrainingExample>,
    pub corrections: Vec<CorrectionRecord>,
    pub iteration:   usize,

    // Statystyki
    pub intent_accuracy:  HashMap<String, (usize, usize)>, // intent -> (ok, total)
    pub concept_hits:     HashMap<String, usize>,
}

impl TrainingManager {
    pub fn new() -> Self {
        TrainingManager {
            examples:       base_examples(),
            corrections:    Vec::new(),
            iteration:      0,
            intent_accuracy: HashMap::new(),
            concept_hits:   HashMap::new(),
        }
    }

    /// Dodaj własny przykład treningowy
    pub fn add_example(
        &mut self,
        input:    &str,
        intent:   &str,
        output:   &str,
        concepts: Vec<&str>,
    ) {
        self.examples.push(TrainingExample::new(input, intent, output, concepts));
        println!("[TRAINING] Dodano przykład: \"{}\" -> {}", input, intent);
    }

    /// Dodaj korekcję — "ta odpowiedź była zła, powinna być taka"
    pub fn add_correction(
        &mut self,
        input:    &str,
        bad:      &str,
        correct:  &str,
        intent:   &str,
    ) {
        self.corrections.push(CorrectionRecord {
            input:            input.to_string(),
            bad_response:     bad.to_string(),
            correct_response: correct.to_string(),
            detected_intent:  intent.to_string(),
            timestamp:        self.iteration,
        });
        println!("[TRAINING] Korekcja zarejestrowana dla: \"{}\"", input);
    }

    /// Oceń odpowiedź względem przykładów treningowych
    pub fn evaluate(
        &mut self,
        input:           &str,
        detected_intent: &str,
        response:        &str,
    ) -> TrainingEval {
        self.iteration += 1;
        let input_lower    = input.to_lowercase();
        let response_lower = response.to_lowercase();

        // Znajdź najbliższy przykład — sklonuj dane żeby zwolnić pożyczkę
        let best_example_data = self.find_best_example(&input_lower).map(|ex| (
            ex.input.clone(),
            ex.expected_intent.clone(),
            ex.expected_output.clone(),
            ex.key_concepts.clone(),
        ));

        let mut eval = TrainingEval {
            score:            0.0,
            intent_correct:   false,
            concepts_found:   Vec::new(),
            concepts_missing: Vec::new(),
            suggestion:       None,
            matched_example:  None,
        };

        if let Some((ex_input, ex_intent, ex_output, ex_concepts)) = best_example_data {
            eval.matched_example = Some(ex_input);

            // Sprawdź intencję
            eval.intent_correct = detected_intent.contains(&ex_intent)
                || ex_intent.contains(detected_intent.replace("INTENT:", "").as_str());

            // Sprawdź pojęcia kluczowe
            for concept in &ex_concepts {
                if response_lower.contains(concept.as_str())
                || input_lower.contains(concept.as_str()) {
                    eval.concepts_found.push(concept.clone());
                    // Teraz możemy mutably pożyczyć self bo ex_concepts jest sklonowane
                    *self.concept_hits.entry(concept.clone()).or_insert(0) += 1;
                } else {
                    eval.concepts_missing.push(concept.clone());
                }
            }

            // Oblicz score
            let intent_score   = if eval.intent_correct { 0.4 } else { 0.0 };
            let total_concepts = ex_concepts.len().max(1) as f32;
            let concept_score  = (eval.concepts_found.len() as f32 / total_concepts) * 0.4;
            let length_score   = if response.len() > 20 { 0.2 } else { 0.05 };

            eval.score = intent_score + concept_score + length_score;

            // Sugestia jeśli słabo
            if eval.score < 0.5 {
                eval.suggestion = Some(ex_output);
            }
        } else {
            // Brak przykładu — podstawowa ocena
            eval.score = if response.len() > 20 { 0.3 } else { 0.1 };
        }

        // Aktualizuj statystyki intencji
        let entry = self.intent_accuracy
            .entry(detected_intent.to_string())
            .or_insert((0, 0));
        entry.1 += 1;
        if eval.intent_correct { entry.0 += 1; }

        eval
    }

    /// Znajdź najlepiej pasujący przykład treningowy
    fn find_best_example(&self, input_lower: &str) -> Option<&TrainingExample> {
        let mut best_score = 0.0_f32;
        let mut best:      Option<&TrainingExample> = None;

        for example in &self.examples {
            let example_lower = example.input.to_lowercase();

            // Policz wspólne słowa
            let input_words:   Vec<&str> = input_lower.split_whitespace().collect();
            let example_words: Vec<&str> = example_lower.split_whitespace().collect();

            let common = input_words.iter()
                .filter(|w| example_words.contains(w))
                .count();

            let score = common as f32
                / (input_words.len().max(example_words.len())) as f32
                * example.weight;

            if score > best_score {
                best_score = score;
                best       = Some(example);
            }
        }

        // Tylko jeśli dopasowanie jest wystarczające
        if best_score > 0.15 { best } else { None }
    }

    /// Generuj odpowiedź na podstawie przykładów (fallback gdy reguły nie matchują)
    pub fn generate_from_examples(&self, input: &str) -> Option<String> {
        let input_lower = input.to_lowercase();
        let example     = self.find_best_example(&input_lower)?;

        // Nie zwracaj wprost — zmodyfikuj odpowiedź
        let base    = &example.expected_output;
        let words:   Vec<&str> = input_lower.split_whitespace().collect();

        // Znajdź słowa kluczowe z wejścia których nie ma w przykładzie
        let new_words: Vec<&&str> = words.iter()
            .filter(|w| {
                w.len() > 3
                && !example.input.to_lowercase().contains(**w)
                && !["what","that","this","with","have","from","they","will","your","been"].contains(*w)
            })
            .collect();

        if new_words.is_empty() {
            Some(base.clone())
        } else {
            // Wzbogać odpowiedź o nowe słowa
            Some(format!("{} (context: {})", base, new_words.iter().map(|w| **w).collect::<Vec<_>>().join(", ")))
        }
    }

    /// Zapisz korekcje do pliku
    pub fn save_corrections(&self, path: &str) {
        let mut out = String::from("# AURORA CORRECTIONS LOG\n\n");
        for (i, c) in self.corrections.iter().enumerate() {
            out.push_str(&format!(
                "[{}] iter={}\nINPUT:   {}\nBAD:     {}\nCORRECT: {}\nINTENT:  {}\n\n",
                i, c.timestamp, c.input, c.bad_response, c.correct_response, c.detected_intent
            ));
        }
        let _ = fs::write(path, out);
    }

    /// Podsumowanie statystyk
    pub fn stats_summary(&self) -> String {
        let total_examples = self.examples.len();
        let total_corrections = self.corrections.len();

        let intent_stats: Vec<String> = self.intent_accuracy.iter()
            .map(|(intent, (ok, total))| {
                format!("{}:{}/{}", intent.replace("INTENT:", ""), ok, total)
            })
            .collect();

        format!(
            "Examples:{} Corrections:{} Intents:[{}] Iter:{}",
            total_examples,
            total_corrections,
            intent_stats.join(" "),
            self.iteration,
        )
    }
}

// ========================
// WYNIK OCENY
// ========================

#[derive(Debug)]
pub struct TrainingEval {
    pub score:            f32,
    pub intent_correct:   bool,
    pub concepts_found:   Vec<String>,
    pub concepts_missing: Vec<String>,
    pub suggestion:       Option<String>,   // lepsza odpowiedź jeśli score < 0.5
    pub matched_example:  Option<String>,
}

impl TrainingEval {
    pub fn print_debug(&self) {
        println!("[EVAL] score={:.3} intent_ok={} concepts={}/{} matched={:?}",
            self.score,
            self.intent_correct,
            self.concepts_found.len(),
            self.concepts_found.len() + self.concepts_missing.len(),
            self.matched_example,
        );
        if !self.concepts_missing.is_empty() {
            println!("[EVAL] missing concepts: {:?}", self.concepts_missing);
        }
        if let Some(s) = &self.suggestion {
            println!("[EVAL] suggestion: \"{}\"", &s[..s.len().min(60)]);
        }
    }
}