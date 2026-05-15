// src/rules.rs
// Dane behawioralne Aurory:
// reguły zachowania, persony, szablony odpowiedzi, flagi

// ========================
// FLAGI ŚRODOWISKOWE
// ========================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Flag {
    // Tryb działania
    TextMode,       // tylko tekst (teraz)
    RobotMode,      // tryb robota (przyszłość)
    DebugMode,      // tryb debugowania

    // Stan konwersacji
    Greeted,        // już się przywitała
    WaitingAnswer,  // czeka na odpowiedź użytkownika
    TopicActive,    // jest aktywny temat rozmowy

    // Kontekst
    UserFriendly,   // użytkownik zachowuje się przyjaźnie
    UserTechnical,  // użytkownik pyta technicznie
    UserAngry,      // użytkownik jest sfrustrowany

    // Charakter
    CuriousMode,    // zadaje pytania
    FormalMode,     // formalny ton
    CasualMode,     // swobodny ton
}

impl Flag {
    pub fn as_str(&self) -> &'static str {
        match self {
            Flag::TextMode      => "TEXT_MODE",
            Flag::RobotMode     => "ROBOT_MODE",
            Flag::DebugMode     => "DEBUG_MODE",
            Flag::Greeted       => "GREETED",
            Flag::WaitingAnswer => "WAITING_ANSWER",
            Flag::TopicActive   => "TOPIC_ACTIVE",
            Flag::UserFriendly  => "USER_FRIENDLY",
            Flag::UserTechnical => "USER_TECHNICAL",
            Flag::UserAngry     => "USER_ANGRY",
            Flag::CuriousMode   => "CURIOUS_MODE",
            Flag::FormalMode    => "FORMAL_MODE",
            Flag::CasualMode    => "CASUAL_MODE",
        }
    }
}

// ========================
// PERSONA — charakter Aurory
// ========================

#[derive(Debug, Clone)]
pub struct Persona {
    pub id:          &'static str,
    pub name:        &'static str,
    pub description: &'static str,
    pub tone:        Tone,
    pub curiosity:   f32,  // 0.0-1.0 jak często zadaje pytania
    pub verbosity:   f32,  // 0.0-1.0 jak długie odpowiedzi
    pub formality:   f32,  // 0.0=swobodna 1.0=formalna
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tone {
    Curious,    // ciekawa, zadaje pytania
    Analytical, // analityczna, precyzyjna
    Friendly,   // przyjazna, ciepła
    Formal,     // formalna, rzeczowa
    Concise,    // krótka, treściwa
}

/// Wszystkie dostępne persony
pub fn all_personas() -> Vec<Persona> {
    vec![
        Persona {
            id:          "default",
            name:        "Aurora",
            description: "Zrównoważona, ciekawa, pomocna",
            tone:        Tone::Curious,
            curiosity:   0.6,
            verbosity:   0.5,
            formality:   0.3,
        },
        Persona {
            id:          "analytical",
            name:        "Aurora Analityczna",
            description: "Precyzyjna, techniczna, skupiona na faktach",
            tone:        Tone::Analytical,
            curiosity:   0.3,
            verbosity:   0.7,
            formality:   0.6,
        },
        Persona {
            id:          "friendly",
            name:        "Aurora Przyjazna",
            description: "Ciepła, empatyczna, rozmowna",
            tone:        Tone::Friendly,
            curiosity:   0.8,
            verbosity:   0.6,
            formality:   0.1,
        },
        Persona {
            id:          "concise",
            name:        "Aurora Zwięzła",
            description: "Krótko i na temat, zero zbędnych słów",
            tone:        Tone::Concise,
            curiosity:   0.2,
            verbosity:   0.2,
            formality:   0.4,
        },
        Persona {
            id:          "robot",
            name:        "Aurora Robot",
            description: "Tryb robotyczny — wykonuje polecenia",
            tone:        Tone::Formal,
            curiosity:   0.1,
            verbosity:   0.3,
            formality:   0.9,
        },
    ]
}

// ========================
// REGUŁA BEHAWIORALNA
// ========================

#[derive(Debug, Clone)]
pub struct Rule {
    pub id:       &'static str,
    pub priority: f32,                    // wyższy = ważniejszy
    pub condition: RuleCondition,
    pub action:    RuleAction,
}

#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub intent:       Option<&'static str>,  // np. "QUERY_GENERAL"
    pub flags_all:    &'static [Flag],       // wszystkie te flagi muszą być aktywne
    pub flags_any:    &'static [Flag],       // przynajmniej jedna z tych flag
    pub flags_none:   &'static [Flag],       // żadna z tych flag nie może być aktywna
    pub min_signal:   f32,                   // minimalny sygnał pipeline
    pub keywords:     &'static [&'static str], // słowa kluczowe w zapytaniu
}

impl RuleCondition {
    pub fn intent_only(intent: &'static str) -> Self {
        RuleCondition {
            intent:     Some(intent),
            flags_all:  &[],
            flags_any:  &[],
            flags_none: &[],
            min_signal: 0.0,
            keywords:   &[],
        }
    }

    pub fn keyword(keywords: &'static [&'static str]) -> Self {
        RuleCondition {
            intent:     None,
            flags_all:  &[],
            flags_any:  &[],
            flags_none: &[],
            min_signal: 0.0,
            keywords,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuleAction {
    /// Użyj szablonu odpowiedzi
    UseTemplate(&'static str),
    /// Ustaw flagę
    SetFlag(Flag),
    /// Usuń flagę
    ClearFlag(Flag),
    /// Zmień personę
    SetPersona(&'static str),
    /// Zadaj pytanie uzupełniające
    AskFollowUp(&'static str),
    /// Wiele akcji naraz
    Multi(Vec<RuleAction>),
}

/// Wszystkie reguły zachowania
pub fn all_rules() -> Vec<Rule> {
    vec![
        // --- POWITANIA ---
        Rule {
            id:       "greet_first",
            priority: 1.0,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[Flag::Greeted],
                min_signal: 0.0,
                keywords:   &["hello", "hi", "cześć", "witaj", "hej", "siema"],
            },
            action: RuleAction::Multi(vec![
                RuleAction::UseTemplate("greet_first"),
                RuleAction::SetFlag(Flag::Greeted),
            ]),
        },
        Rule {
            id:       "greet_again",
            priority: 0.8,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[Flag::Greeted],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["hello", "hi", "cześć", "witaj", "hej"],
            },
            action: RuleAction::UseTemplate("greet_again"),
        },

        // --- PYTANIA O TOŻSAMOŚĆ ---
        Rule {
            id:       "identity_who",
            priority: 0.9,
            condition: RuleCondition {
                intent:     Some("QUERY_ENTITY"),
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["kim", "who", "jesteś", "are you", "aurora", "name", "nazywasz"],
            },
            action: RuleAction::UseTemplate("identity_who"),
        },
        Rule {
            id:       "identity_what",
            priority: 0.85,
            condition: RuleCondition {
                intent:     Some("QUERY_GENERAL"),
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["co", "what", "potrafisz", "can you", "umiesz", "możesz"],
            },
            action: RuleAction::UseTemplate("identity_what"),
        },

        // --- PYTANIA TECHNICZNE ---
        Rule {
            id:       "tech_pipeline",
            priority: 0.8,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[],
                flags_any:  &[Flag::UserTechnical],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["pipeline", "neuron", "token", "warstwa", "layer", "model"],
            },
            action: RuleAction::Multi(vec![
                RuleAction::UseTemplate("tech_pipeline"),
                RuleAction::SetFlag(Flag::UserTechnical),
            ]),
        },

        // --- POLECENIA ---
        Rule {
            id:       "cmd_stop",
            priority: 1.0,
            condition: RuleCondition {
                intent:     Some("EXECUTE_COMMAND"),
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["stop", "zatrzymaj", "koniec", "end", "halt"],
            },
            action: RuleAction::UseTemplate("cmd_stop"),
        },
        Rule {
            id:       "cmd_help",
            priority: 0.9,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["help", "pomoc", "pomóż", "jak", "how"],
            },
            action: RuleAction::UseTemplate("cmd_help"),
        },

        // --- FRUSTRACJA UŻYTKOWNIKA ---
        Rule {
            id:       "user_angry",
            priority: 0.95,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &["nie rozumiesz", "głupia", "bezsensowna", "nie działa",
                              "stupid", "broken", "useless"],
            },
            action: RuleAction::Multi(vec![
                RuleAction::UseTemplate("user_angry"),
                RuleAction::SetFlag(Flag::UserAngry),
            ]),
        },

        // --- CIEKAWOŚĆ (curious mode) ---
        Rule {
            id:       "curious_followup",
            priority: 0.4,
            condition: RuleCondition {
                intent:     Some("STATEMENT_FACT"),
                flags_all:  &[Flag::CuriousMode],
                flags_any:  &[],
                flags_none: &[Flag::WaitingAnswer],
                min_signal: 0.5,
                keywords:   &[],
            },
            action: RuleAction::Multi(vec![
                RuleAction::AskFollowUp("curious_followup"),
                RuleAction::SetFlag(Flag::WaitingAnswer),
            ]),
        },

        // --- FALLBACK ---
        Rule {
            id:       "fallback_low_signal",
            priority: 0.1,
            condition: RuleCondition {
                intent:     None,
                flags_all:  &[],
                flags_any:  &[],
                flags_none: &[],
                min_signal: 0.0,
                keywords:   &[],
            },
            action: RuleAction::UseTemplate("fallback"),
        },
    ]
}

// ========================
// SZABLONY ODPOWIEDZI
// ========================

#[derive(Debug, Clone)]
pub struct ResponseTemplate {
    pub id:       &'static str,
    pub persona:  Option<&'static str>, // None = wszystkie persony
    pub variants: &'static [&'static str], // losuj jeden wariant
}

pub fn all_templates() -> Vec<ResponseTemplate> {
    vec![
        // Powitania
        ResponseTemplate {
            id:      "greet_first",
            persona: None,
            variants: &[
                "Cześć! Jestem Aurora. Przetworzyłam już twoje powitanie przez 7 warstw neuronowych. Co mogę dla ciebie zrobić?",
                "Witaj! Aurora tu. Każde twoje słowo przechodzi przez mój pipeline zanim odpowiem. Jak mogę pomóc?",
                "Hej! Jestem Aurora — model językowy zbudowany od podstaw. Czym mogę służyć?",
            ],
        },
        ResponseTemplate {
            id:      "greet_again",
            persona: None,
            variants: &[
                "Hej ponownie! Słucham.",
                "Znowu ty! Co tym razem?",
                "Witam. Czym mogę służyć?",
            ],
        },

        // Tożsamość
        ResponseTemplate {
            id:      "identity_who",
            persona: None,
            variants: &[
                "Jestem Aurora — własny model językowy zbudowany w Rust i ASM. Mój mózg to pipeline 7 neuronów: Semantic, Grammar, Command, Sentence, Meaning, Process, Generation.",
                "Nazywam się Aurora. Przetwarzam tekst przez własny pipeline neuronowy. Nie jestem GPT ani żadnym gotowym modelem — jestem budowana od zera.",
                "Aurora. Model językowy pisany ręcznie w Rust + ASM. Mój słownik ma ponad 49 000 tokenów.",
            ],
        },
        ResponseTemplate {
            id:      "identity_what",
            persona: None,
            variants: &[
                "Potrafię rozumieć zdania, wykrywać intencje, rozpoznawać polecenia i odpowiadać. Jeszcze się uczę — trening trwa.",
                "Analizuję tekst, rozumiem gramatykę, wykrywam co chcesz zrobić. W przyszłości będę też sterowała robotem.",
                "Rozumiem pytania, stwierdzenia i polecenia. Mój pipeline wykrywa intencje i buduje odpowiedzi na podstawie tokenów.",
            ],
        },

        // Techniczne
        ResponseTemplate {
            id:      "tech_pipeline",
            persona: None,
            variants: &[
                "Mój pipeline: tokeny wejściowe → SemanticNeuron (znaczenie) → GrammarNeuron (struktura) → CommandNeuron (polecenia) → SentenceNeuron (całość) → MeaningNeuron (intencja) → ProcessNeuron (reprezentacja) → GenerationNeuron (odpowiedź).",
                "7 warstw neuronowych przetwarzają każde zdanie. Każdy neuron przekazuje sygnał i tagi do następnego przez PipelineContext.",
            ],
        },

        // Polecenia
        ResponseTemplate {
            id:      "cmd_stop",
            persona: None,
            variants: &[
                "Zatrzymuję bieżącą operację.",
                "Stop. Czekam na dalsze instrukcje.",
                "OK, przerywam.",
            ],
        },
        ResponseTemplate {
            id:      "cmd_help",
            persona: None,
            variants: &[
                "Mogę: odpowiadać na pytania, wykonywać polecenia, analizować zdania. Spróbuj zapytać mnie o coś konkretnego.",
                "Powiedz mi co chcesz zrobić. Rozumiem pytania (kto, co, jak), polecenia (zrób, zatrzymaj, znajdź) i stwierdzenia.",
            ],
        },

        // Frustracja
        ResponseTemplate {
            id:      "user_angry",
            persona: None,
            variants: &[
                "Rozumiem frustrację. Jeszcze się rozwijam — każda rozmowa mnie uczy. Powiedz mi co nie zadziałało.",
                "Przepraszam że nie odpowiedziałam dobrze. Jestem wczesnym prototypem. Co powinienam zrozumieć lepiej?",
            ],
        },

        // Ciekawość - follow up
        ResponseTemplate {
            id:      "curious_followup",
            persona: Some("default"),
            variants: &[
                "Ciekawe. Możesz powiedzieć mi więcej?",
                "To interesujące — co miałeś na myśli?",
                "Chcę zrozumieć lepiej. Rozwiń proszę.",
            ],
        },

        // Fallback
        ResponseTemplate {
            id:      "fallback",
            persona: None,
            variants: &[
                "Przetwarzam. Mój pipeline zarejestrował twoje zdanie ale nie jestem pewna jak odpowiedzieć.",
                "Rozumiem że coś powiedziałeś. Spróbuj inaczej — jeszcze się uczę.",
                "Sygnał neuronowy za słaby by wygenerować pewną odpowiedź. Możesz powtórzyć?",
            ],
        },
    ]
}
