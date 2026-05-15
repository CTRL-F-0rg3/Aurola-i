// src/neuron.rs
// Jeden uniwersalny Neuron — inicjowany w różnych konfiguracjach
// Pipeline: Semantic -> Grammar -> Command -> Sentence -> Meaning -> Process -> Generation

use std::collections::HashMap;

// ========================
// TYPY PODSTAWOWE
// ========================

pub type TokenId  = u32;
pub type PosId    = u32;
pub type Weight   = f32;
pub type Signal   = f32;

/// Pojedynczy token z pełnym kontekstem
#[derive(Debug, Clone)]
pub struct TokenContext {
    pub id:       TokenId,
    pub word:     String,
    pub pos_id:   PosId,
    pub lemma_id: TokenId,
    pub position: usize,     // pozycja w zdaniu
}

/// Wynik przetwarzania przez neuron
#[derive(Debug, Clone)]
pub struct NeuronOutput {
    pub signal:    Signal,           // główny sygnał wyjściowy 0.0-1.0
    pub tags:      Vec<String>,      // tagi semantyczne/gramatyczne
    pub data:      HashMap<String, f32>, // dodatkowe dane numeryczne
    pub tokens:    Vec<TokenId>,     // tokeny wyjściowe (dla GenerationNeuron)
    pub fired:     bool,             // czy neuron "wystrzelił"
}

impl NeuronOutput {
    pub fn empty() -> Self {
        NeuronOutput {
            signal: 0.0,
            tags:   Vec::new(),
            data:   HashMap::new(),
            tokens: Vec::new(),
            fired:  false,
        }
    }

    pub fn fired(signal: Signal) -> Self {
        NeuronOutput {
            signal,
            tags:   Vec::new(),
            data:   HashMap::new(),
            tokens: Vec::new(),
            fired:  true,
        }
    }
}

// ========================
// TYP NEURONU
// ========================

#[derive(Debug, Clone, PartialEq)]
pub enum NeuronType {
    Semantic,    // [1] znaczenie słowa
    Grammar,     // [2] stan gramatyczny zdania
    Command,     // [3] detekcja poleceń
    Sentence,    // [4] analiza całego zdania
    Meaning,     // [5] przypisanie sensu/intencji
    Process,     // [6] przetworzenie sensu
    Generation,  // [7] generacja odpowiedzi
}

impl NeuronType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NeuronType::Semantic   => "SEMANTIC",
            NeuronType::Grammar    => "GRAMMAR",
            NeuronType::Command    => "COMMAND",
            NeuronType::Sentence   => "SENTENCE",
            NeuronType::Meaning    => "MEANING",
            NeuronType::Process    => "PROCESS",
            NeuronType::Generation => "GENERATION",
        }
    }
}

// ========================
// FUNKCJE AKTYWACJI
// ========================

#[derive(Debug, Clone)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Threshold(f32), // odpala tylko jeśli > próg
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU           => x.max(0.0),
            Activation::Sigmoid        => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh           => x.tanh(),
            Activation::Linear         => x,
            Activation::Threshold(t)   => if x > *t { x } else { 0.0 },
        }
    }
}

// ========================
// POŁĄCZENIE MIĘDZY NEURONAMI
// ========================

#[derive(Debug, Clone)]
pub struct Synapse {
    pub target_id: usize,   // ID neuronu docelowego
    pub weight:    Weight,  // waga połączenia
    pub inhibitory: bool,   // hamujące (true) lub pobudzające (false)
}

impl Synapse {
    pub fn excite(target_id: usize, weight: Weight) -> Self {
        Synapse { target_id, weight, inhibitory: false }
    }

    pub fn inhibit(target_id: usize, weight: Weight) -> Self {
        Synapse { target_id, weight, inhibitory: true }
    }

    pub fn effective_weight(&self) -> Weight {
        if self.inhibitory { -self.weight } else { self.weight }
    }
}

// ========================
// GŁÓWNA STRUKTURA NEURONU
// ========================

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id:          usize,
    pub neuron_type: NeuronType,
    pub activation:  Activation,
    pub threshold:   f32,        // próg aktywacji
    pub bias:        f32,        // bias
    pub synapses:    Vec<Synapse>, // połączenia wyjściowe
    pub state:       f32,        // aktualny stan wewnętrzny
    pub last_output: NeuronOutput,

    // Dane specyficzne dla typu neuronu
    pub semantic_vocab:  HashMap<TokenId, Vec<String>>, // token -> tagi semantyczne
    pub grammar_rules:   Vec<GrammarRule>,
    pub command_patterns: Vec<CommandPattern>,
    pub memory:          Vec<NeuronOutput>, // krótka pamięć ostatnich wyjść
    pub memory_size:     usize,
}

impl Neuron {
    /// Tworzy nowy neuron danego typu
    pub fn new(id: usize, neuron_type: NeuronType) -> Self {
        let (activation, threshold) = match &neuron_type {
            NeuronType::Semantic   => (Activation::Sigmoid,        0.3),
            NeuronType::Grammar    => (Activation::Threshold(0.5), 0.5),
            NeuronType::Command    => (Activation::Threshold(0.7), 0.7),
            NeuronType::Sentence   => (Activation::Tanh,           0.4),
            NeuronType::Meaning    => (Activation::Sigmoid,        0.4),
            NeuronType::Process    => (Activation::ReLU,           0.2),
            NeuronType::Generation => (Activation::Sigmoid,        0.3),
        };

        Neuron {
            id,
            neuron_type,
            activation,
            threshold,
            bias: 0.1,
            synapses:         Vec::new(),
            state:            0.0,
            last_output:      NeuronOutput::empty(),
            semantic_vocab:   HashMap::new(),
            grammar_rules:    Vec::new(),
            command_patterns: Vec::new(),
            memory:           Vec::new(),
            memory_size:      8,
        }
    }

    /// Połącz ten neuron z innym (dodaj synapsę)
    pub fn connect_to(&mut self, target_id: usize, weight: Weight) {
        self.synapses.push(Synapse::excite(target_id, weight));
    }

    pub fn inhibit(&mut self, target_id: usize, weight: Weight) {
        self.synapses.push(Synapse::inhibit(target_id, weight));
    }

    /// Zapisz wynik do pamięci krótkotrwałej
    fn remember(&mut self, output: NeuronOutput) {
        if self.memory.len() >= self.memory_size {
            self.memory.remove(0);
        }
        self.memory.push(output);
    }

    /// Główna funkcja przetwarzania — zależna od typu neuronu
    pub fn process(
        &mut self,
        tokens:   &[TokenContext],
        input:    Signal,
        context:  &PipelineContext,
    ) -> NeuronOutput {

        let raw = match &self.neuron_type {
            NeuronType::Semantic   => self.process_semantic(tokens, input, context),
            NeuronType::Grammar    => self.process_grammar(tokens, input, context),
            NeuronType::Command    => self.process_command(tokens, input, context),
            NeuronType::Sentence   => self.process_sentence(tokens, input, context),
            NeuronType::Meaning    => self.process_meaning(tokens, input, context),
            NeuronType::Process    => self.process_process(tokens, input, context),
            NeuronType::Generation => self.process_generation(tokens, input, context),
        };

        // Aktualizuj stan wewnętrzny
        self.state = self.activation.apply(raw.signal + self.bias);

        let mut output = raw;
        output.signal = self.state;
        output.fired  = self.state > self.threshold;

        self.remember(output.clone());
        self.last_output = output.clone();
        output
    }

    // ========================
    // [1] SEMANTIC NEURON
    // Cel: zrozumieć znaczenie każdego tokenu
    // ========================
    fn process_semantic(
        &mut self,
        tokens:  &[TokenContext],
        _input:  Signal,
        _ctx:    &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        for token in tokens {
            // Sprawdź tagi semantyczne z vocab
            if let Some(tags) = self.semantic_vocab.get(&token.id) {
                for tag in tags {
                    if !out.tags.contains(tag) {
                        out.tags.push(tag.clone());
                    }
                }
            }

            // Kategoryzacja na podstawie POS
            let sem_tag = match token.pos_id {
                1  => "ENTITY",      // N  -> encja
                2  => "ACTION",      // V  -> akcja
                3  => "PROPERTY",    // ADJ-> właściwość
                4  => "MODIFIER",    // ADV-> modyfikator
                5  => "REFERENCE",   // PRN-> referencja
                6  => "RELATION",    // PRP-> relacja
                7  => "CONNECTOR",   // CNJ-> łącznik
                8  => "QUANTITY",    // NUM-> ilość
                10 => "PROPER_NAME", // NP -> nazwa własna
                11 => "DETERMINER",  // DET-> determinator
                16 => "AUXILIARY",   // AUX-> posiłkowy
                17 => "MODAL",       // MOD-> modalny
                _  => "UNKNOWN",
            };

            if !out.tags.contains(&sem_tag.to_string()) {
                out.tags.push(sem_tag.to_string());
            }

            // Sygnał: im więcej rozpoznanych tokenów tym wyższy
            out.signal += 1.0 / tokens.len() as f32;
        }

        out.data.insert("token_count".into(), tokens.len() as f32);
        out.data.insert("unique_pos".into(),
            tokens.iter().map(|t| t.pos_id).collect::<std::collections::HashSet<_>>().len() as f32
        );
        out
    }

    // ========================
    // [2] GRAMMAR NEURON
    // Cel: określić strukturę gramatyczną zdania
    // ========================
    fn process_grammar(
        &mut self,
        tokens:  &[TokenContext],
        _input:  Signal,
        _ctx:    &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        let pos_sequence: Vec<PosId> = tokens.iter().map(|t| t.pos_id).collect();

        // Sprawdź każdą regułę gramatyczną
        let mut matched_rules = Vec::new();
        for rule in &self.grammar_rules {
            if rule.matches(&pos_sequence) {
                matched_rules.push(rule.name.clone());
                out.signal += rule.confidence;
            }
        }

        // Wykryj podstawowe wzorce
        let has_verb    = pos_sequence.contains(&2);
        let has_noun    = pos_sequence.contains(&1);
        let has_det     = pos_sequence.contains(&11);
        let has_modal   = pos_sequence.contains(&17);
        let has_aux     = pos_sequence.contains(&16);

        if has_verb && has_noun {
            out.tags.push("COMPLETE_CLAUSE".into());
            out.signal += 0.4;
        }
        if has_modal || has_aux {
            out.tags.push("COMPLEX_VERB".into());
            out.signal += 0.1;
        }
        if !has_verb {
            out.tags.push("FRAGMENT".into());
        }
        if has_det && has_noun {
            out.tags.push("HAS_NP".into());
        }

        // Wykryj typ zdania
        let sentence_type = self.detect_sentence_type(&pos_sequence, tokens);
        out.tags.push(sentence_type.clone());
        out.data.insert("sentence_type_hash".into(), sentence_type.len() as f32);

        for rule in matched_rules {
            out.tags.push(format!("RULE:{}", rule));
        }

        out.signal = (out.signal / tokens.len().max(1) as f32).clamp(0.0, 1.0);
        out
    }

    fn detect_sentence_type(&self, pos_seq: &[PosId], tokens: &[TokenContext]) -> String {
        // Pytanie: zaczyna się od AUX/MOD lub kończy na ?
        let starts_aux = pos_seq.first().map(|&p| p == 16 || p == 17).unwrap_or(false);
        let has_question_word = tokens.iter().any(|t| {
            matches!(t.word.to_uppercase().as_str(), "WHO"|"WHAT"|"WHERE"|"WHEN"|"WHY"|"HOW"|"WHICH")
        });
        let last_is_punct = tokens.last().map(|t| t.word == "?").unwrap_or(false);

        if starts_aux || has_question_word || last_is_punct {
            return "QUESTION".into();
        }

        // Polecenie: zaczyna się od czasownika bez podmiotu
        let starts_verb = pos_seq.first().map(|&p| p == 2).unwrap_or(false);
        if starts_verb {
            return "IMPERATIVE".into();
        }

        "STATEMENT".into()
    }

    // ========================
    // [3] COMMAND NEURON
    // Cel: wykryć czy zdanie to polecenie
    // ========================
    fn process_command(
        &mut self,
        tokens:  &[TokenContext],
        _input:  Signal,
        ctx:     &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        // Sprawdź czy gramatyka wykryła IMPERATIVE
        if ctx.grammar_tags.contains(&"IMPERATIVE".to_string()) {
            out.signal += 0.6;
            out.tags.push("IS_COMMAND".into());
        }

        if ctx.grammar_tags.contains(&"QUESTION".to_string()) {
            out.signal += 0.3;
            out.tags.push("IS_QUESTION".into());
        }

        // Sprawdź wzorce poleceń
        for pattern in &self.command_patterns {
            if pattern.matches(tokens) {
                out.tags.push(format!("CMD:{}", pattern.name));
                out.data.insert(format!("cmd_{}", pattern.name), pattern.priority);
                out.signal += 0.3 * pattern.priority;
            }
        }

        // Słowa kluczowe poleceń
        let command_words = [
            "RUN", "STOP", "START", "MOVE", "GO", "FIND",
            "SEARCH", "OPEN", "CLOSE", "SET", "GET", "LIST",
            "SHOW", "HELP", "TELL", "SAY", "DO", "MAKE",
        ];

        for token in tokens {
            if command_words.contains(&token.word.to_uppercase().as_str()) {
                out.tags.push("HAS_COMMAND_WORD".into());
                out.signal += 0.2;
                break;
            }
        }

        out.signal = out.signal.clamp(0.0, 1.0);
        out
    }

    // ========================
    // [4] SENTENCE NEURON
    // Cel: analiza zdania jako całości
    // ========================
    fn process_sentence(
        &mut self,
        tokens:  &[TokenContext],
        _input:  Signal,
        ctx:     &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        // Zbierz informacje z poprzednich warstw
        let sem_signal  = ctx.semantic_signal;
        let gram_signal = ctx.grammar_signal;

        // Połącz sygnały
        out.signal = (sem_signal * 0.4 + gram_signal * 0.6).clamp(0.0, 1.0);

        // Wykryj podmiot i orzeczenie
        let mut subject_idx:   Option<usize> = None;
        let mut predicate_idx: Option<usize> = None;
        let mut object_idx:    Option<usize> = None;

        for (i, token) in tokens.iter().enumerate() {
            match token.pos_id {
                1 | 5 if subject_idx.is_none() => {
                    subject_idx = Some(i);
                    out.tags.push(format!("SUBJECT:{}", token.word));
                }
                2 if predicate_idx.is_none() => {
                    predicate_idx = Some(i);
                    out.tags.push(format!("PREDICATE:{}", token.word));
                }
                1 if subject_idx.is_some() && predicate_idx.is_some() && object_idx.is_none() => {
                    object_idx = Some(i);
                    out.tags.push(format!("OBJECT:{}", token.word));
                }
                _ => {}
            }
        }

        // Kompletność struktury SVO
        let svo_score = match (subject_idx, predicate_idx, object_idx) {
            (Some(_), Some(_), Some(_)) => 1.0,
            (Some(_), Some(_), None)    => 0.7,
            (None,    Some(_), Some(_)) => 0.6,
            (Some(_), None,    _)       => 0.3,
            _                           => 0.1,
        };

        out.data.insert("svo_score".into(), svo_score);
        out.data.insert("length".into(), tokens.len() as f32);
        out.signal = (out.signal * 0.5 + svo_score * 0.5).clamp(0.0, 1.0);

        // Propaguj tagi semantyczne i gramatyczne
        for tag in &ctx.semantic_tags {
            out.tags.push(format!("SEM:{}", tag));
        }
        for tag in &ctx.grammar_tags {
            out.tags.push(format!("GRAM:{}", tag));
        }

        out
    }

    // ========================
    // [5] MEANING NEURON
    // Cel: przypisanie intencji/sensu
    // ========================
    fn process_meaning(
        &mut self,
        _tokens: &[TokenContext],
        _input:  Signal,
        ctx:     &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        // Intencja na podstawie tagów
        let intent = if ctx.command_tags.contains(&"IS_QUESTION".to_string()) {
            if ctx.semantic_tags.contains(&"PROPER_NAME".to_string()) {
                "QUERY_ENTITY"
            } else if ctx.semantic_tags.contains(&"ACTION".to_string()) {
                "QUERY_ACTION"
            } else {
                "QUERY_GENERAL"
            }
        } else if ctx.command_tags.contains(&"IS_COMMAND".to_string()) {
            if ctx.command_tags.contains(&"HAS_COMMAND_WORD".to_string()) {
                "EXECUTE_COMMAND"
            } else {
                "REQUEST_ACTION"
            }
        } else if ctx.semantic_tags.contains(&"ENTITY".to_string())
               && ctx.semantic_tags.contains(&"ACTION".to_string()) {
            "STATEMENT_FACT"
        } else {
            "STATEMENT_GENERAL"
        };

        out.tags.push(format!("INTENT:{}", intent));
        out.data.insert("intent_confidence".into(), ctx.sentence_signal);

        // Sentyment (uproszczony)
        let sentiment = if ctx.semantic_tags.contains(&"MODIFIER".to_string()) {
            0.6 // modyfikatory mogą zmieniać sentiment
        } else {
            0.5 // neutralny
        };
        out.data.insert("sentiment".into(), sentiment);

        out.signal = ctx.sentence_signal * 0.8 + 0.2;
        out.signal = out.signal.clamp(0.0, 1.0);
        out
    }

    // ========================
    // [6] PROCESS NEURON
    // Cel: przetworzenie sensu w reprezentację wewnętrzną
    // ========================
    fn process_process(
        &mut self,
        tokens:  &[TokenContext],
        _input:  Signal,
        ctx:     &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        // Zbuduj wewnętrzną reprezentację zdania
        let intent = ctx.meaning_tags.iter()
            .find(|t| t.starts_with("INTENT:"))
            .cloned()
            .unwrap_or("INTENT:UNKNOWN".into());

        out.tags.push(intent.clone());

        // Wagi tokenów — które są najważniejsze?
        let mut weighted_tokens: Vec<(TokenId, f32)> = Vec::new();
        for token in tokens {
            let weight = match token.pos_id {
                2  => 0.9,  // czasownik — najważniejszy
                1  => 0.8,  // rzeczownik
                10 => 0.85, // nazwa własna
                3  => 0.5,  // przymiotnik
                4  => 0.4,  // przysłówek
                8  => 0.6,  // liczebnik
                _  => 0.2,
            };
            weighted_tokens.push((token.id, weight));
        }

        // Posortuj tokeny po wadze
        weighted_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Zapisz top tokeny jako reprezentację
        for (tid, w) in &weighted_tokens {
            out.data.insert(format!("tok_{}", tid), *w);
        }

        // Główne tokeny do generacji odpowiedzi
        out.tokens = weighted_tokens.iter()
            .take(5)
            .map(|(tid, _)| *tid)
            .collect();

        out.signal = ctx.meaning_signal.clamp(0.0, 1.0);
        out
    }

    // ========================
    // [7] GENERATION NEURON
    // Cel: generacja tokenów odpowiedzi
    // ========================
    fn process_generation(
        &mut self,
        _tokens: &[TokenContext],
        _input:  Signal,
        ctx:     &PipelineContext,
    ) -> NeuronOutput {
        let mut out = NeuronOutput::empty();

        // Pobierz intencję
        let intent = ctx.meaning_tags.iter()
            .find(|t| t.starts_with("INTENT:"))
            .map(|s| s.replace("INTENT:", ""))
            .unwrap_or("UNKNOWN".into());

        // Na podstawie intencji i kluczowych tokenów buduj odpowiedź
        // To jest uproszczone — docelowo tu będą wagi nauczone
        let response_tokens = match intent.as_str() {
            "QUERY_GENERAL" => {
                // Pytanie ogólne — próbuj odpowiedzieć o sobie
                ctx.process_tokens.clone()
            }
            "QUERY_ENTITY" => {
                // Pytanie o encję
                ctx.process_tokens.clone()
            }
            "EXECUTE_COMMAND" => {
                // Polecenie — potwierdź
                ctx.process_tokens.iter().take(2).cloned().collect()
            }
            "STATEMENT_FACT" => {
                // Stwierdzenie — potwierdź lub rozszerz
                ctx.process_tokens.clone()
            }
            _ => {
                ctx.process_tokens.iter().take(3).cloned().collect()
            }
        };

        out.tokens = response_tokens;
        out.tags.push(format!("RESPONSE_FOR:{}", intent));
        out.data.insert("response_confidence".into(), ctx.process_signal);
        out.signal = ctx.process_signal.clamp(0.0, 1.0);
        out.fired  = out.signal > self.threshold;
        out
    }
}

// ========================
// KONTEKST PIPELINE
// Przekazywany między neuronami
// ========================

#[derive(Debug, Clone, Default)]
pub struct PipelineContext {
    // Sygnały z każdej warstwy
    pub semantic_signal:  Signal,
    pub grammar_signal:   Signal,
    pub command_signal:   Signal,
    pub sentence_signal:  Signal,
    pub meaning_signal:   Signal,
    pub process_signal:   Signal,

    // Tagi z każdej warstwy
    pub semantic_tags:    Vec<String>,
    pub grammar_tags:     Vec<String>,
    pub command_tags:     Vec<String>,
    pub sentence_tags:    Vec<String>,
    pub meaning_tags:     Vec<String>,
    pub process_tags:     Vec<String>,

    // Tokeny z Process (do generacji)
    pub process_tokens:   Vec<TokenId>,
}

// ========================
// REGUŁY GRAMATYCZNE
// ========================

#[derive(Debug, Clone)]
pub struct GrammarRule {
    pub name:       String,
    pub pattern:    Vec<PosId>,   // sekwencja POS IDs
    pub confidence: f32,
    pub partial:    bool,         // czy dopasowanie częściowe wystarczy
}

impl GrammarRule {
    pub fn new(name: &str, pattern: Vec<PosId>, confidence: f32) -> Self {
        GrammarRule { name: name.into(), pattern, confidence, partial: true }
    }

    pub fn matches(&self, pos_seq: &[PosId]) -> bool {
        if self.partial {
            // Szukaj wzorca jako podciągu
            if self.pattern.is_empty() || pos_seq.len() < self.pattern.len() {
                return false;
            }
            pos_seq.windows(self.pattern.len())
                .any(|w| w == self.pattern.as_slice())
        } else {
            pos_seq == self.pattern.as_slice()
        }
    }
}

// ========================
// WZORCE POLECEŃ
// ========================

#[derive(Debug, Clone)]
pub struct CommandPattern {
    pub name:     String,
    pub keywords: Vec<String>,
    pub priority: f32,
}

impl CommandPattern {
    pub fn new(name: &str, keywords: Vec<&str>, priority: f32) -> Self {
        CommandPattern {
            name:     name.into(),
            keywords: keywords.iter().map(|s| s.to_uppercase()).collect(),
            priority,
        }
    }

    pub fn matches(&self, tokens: &[TokenContext]) -> bool {
        let words: Vec<String> = tokens.iter()
            .map(|t| t.word.to_uppercase())
            .collect();
        self.keywords.iter().any(|kw| words.contains(kw))
    }
}

// ========================
// SIEĆ NEURONÓW — PIPELINE
// ========================

pub struct NeuralPipeline {
    pub neurons: Vec<Neuron>,
}

impl NeuralPipeline {
    /// Inicjalizuje pełny pipeline 7 neuronów
    pub fn new() -> Self {
        let mut neurons = vec![
            Neuron::new(0, NeuronType::Semantic),
            Neuron::new(1, NeuronType::Grammar),
            Neuron::new(2, NeuronType::Command),
            Neuron::new(3, NeuronType::Sentence),
            Neuron::new(4, NeuronType::Meaning),
            Neuron::new(5, NeuronType::Process),
            Neuron::new(6, NeuronType::Generation),
        ];

        // Połączenia sekwencyjne (każdy z następnym)
        for i in 0..6 {
            neurons[i].connect_to(i + 1, 1.0);
        }

        // Dodatkowe połączenia cross-layer
        neurons[0].connect_to(3, 0.5); // Semantic -> Sentence (bypass)
        neurons[1].connect_to(4, 0.4); // Grammar  -> Meaning
        neurons[2].connect_to(5, 0.6); // Command  -> Process

        // Załaduj podstawowe reguły gramatyczne
        neurons[1].grammar_rules = vec![
            GrammarRule::new("NP_DET_N",   vec![11, 1],    0.8),
            GrammarRule::new("NP_DET_ADJ_N",vec![11, 3, 1],0.9),
            GrammarRule::new("VP_V_NP",    vec![2, 11, 1], 0.8),
            GrammarRule::new("VP_V",       vec![2],         0.5),
            GrammarRule::new("SVO",        vec![1, 2, 1],   1.0),
            GrammarRule::new("SVO_DET",    vec![11, 1, 2, 11, 1], 1.0),
            GrammarRule::new("MODAL_V",    vec![17, 2],     0.7),
            GrammarRule::new("AUX_V",      vec![16, 2],     0.7),
        ];

        // Załaduj wzorce poleceń
        neurons[2].command_patterns = vec![
            CommandPattern::new("MOVE",   vec!["MOVE", "GO", "WALK"],  0.9),
            CommandPattern::new("STOP",   vec!["STOP", "HALT", "END"], 0.9),
            CommandPattern::new("INFO",   vec!["TELL", "SAY", "SHOW", "WHAT", "WHO"], 0.7),
            CommandPattern::new("SEARCH", vec!["FIND", "SEARCH", "LOOK"], 0.8),
            CommandPattern::new("SET",    vec!["SET", "CONFIG", "CHANGE"], 0.8),
        ];

        NeuralPipeline { neurons }
    }

    /// Przepuść tokeny przez cały pipeline
    pub fn run(&mut self, tokens: &[TokenContext]) -> PipelineResult {
        let mut ctx = PipelineContext::default();
        let input_signal = 1.0_f32;

        // [1] Semantic
        let sem = self.neurons[0].process(tokens, input_signal, &ctx);
        ctx.semantic_signal = sem.signal;
        ctx.semantic_tags   = sem.tags.clone();

        // [2] Grammar
        let gram = self.neurons[1].process(tokens, sem.signal, &ctx);
        ctx.grammar_signal = gram.signal;
        ctx.grammar_tags   = gram.tags.clone();

        // [3] Command
        let cmd = self.neurons[2].process(tokens, gram.signal, &ctx);
        ctx.command_signal = cmd.signal;
        ctx.command_tags   = cmd.tags.clone();

        // [4] Sentence
        let sent = self.neurons[3].process(tokens, cmd.signal, &ctx);
        ctx.sentence_signal = sent.signal;
        ctx.sentence_tags   = sent.tags.clone();

        // [5] Meaning
        let mean = self.neurons[4].process(tokens, sent.signal, &ctx);
        ctx.meaning_signal = mean.signal;
        ctx.meaning_tags   = mean.tags.clone();

        // [6] Process
        let proc = self.neurons[5].process(tokens, mean.signal, &ctx);
        ctx.process_signal = proc.signal;
        ctx.process_tags   = proc.tags.clone();
        ctx.process_tokens = proc.tokens.clone();

        // [7] Generation
        let gen = self.neurons[6].process(tokens, proc.signal, &ctx);

        PipelineResult {
            semantic:   sem,
            grammar:    gram,
            command:    cmd,
            sentence:   sent,
            meaning:    mean,
            process:    proc,
            generation: gen,
            context:    ctx,
        }
    }
}

// ========================
// WYNIK CAŁEGO PIPELINE
// ========================

#[derive(Debug)]
pub struct PipelineResult {
    pub semantic:   NeuronOutput,
    pub grammar:    NeuronOutput,
    pub command:    NeuronOutput,
    pub sentence:   NeuronOutput,
    pub meaning:    NeuronOutput,
    pub process:    NeuronOutput,
    pub generation: NeuronOutput,
    pub context:    PipelineContext,
}

impl PipelineResult {
    /// Wypisz podsumowanie pipeline
    pub fn print_summary(&self) {
        println!("=== PIPELINE RESULT ===");
        println!("[1] SEMANTIC   signal={:.3} fired={} tags={:?}",
            self.semantic.signal, self.semantic.fired, self.semantic.tags);
        println!("[2] GRAMMAR    signal={:.3} fired={} tags={:?}",
            self.grammar.signal, self.grammar.fired, self.grammar.tags);
        println!("[3] COMMAND    signal={:.3} fired={} tags={:?}",
            self.command.signal, self.command.fired, self.command.tags);
        println!("[4] SENTENCE   signal={:.3} fired={} tags={:?}",
            self.sentence.signal, self.sentence.fired, self.sentence.tags);
        println!("[5] MEANING    signal={:.3} fired={} tags={:?}",
            self.meaning.signal, self.meaning.fired, self.meaning.tags);
        println!("[6] PROCESS    signal={:.3} fired={} tokens={:?}",
            self.process.signal, self.process.fired, self.process.tokens);
        println!("[7] GENERATION signal={:.3} fired={} response_tokens={:?}",
            self.generation.signal, self.generation.fired, self.generation.tokens);
        println!("========================");
    }

    /// Pobierz główną intencję
    pub fn intent(&self) -> String {
        self.meaning.tags.iter()
            .find(|t| t.starts_with("INTENT:"))
            .cloned()
            .unwrap_or("INTENT:UNKNOWN".into())
    }

    /// Pobierz tokeny do wygenerowania odpowiedzi
    pub fn response_tokens(&self) -> &[TokenId] {
        &self.generation.tokens
    }
}