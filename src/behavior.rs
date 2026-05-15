// src/behavior.rs
// Silnik behawioralny — dopasowuje reguły do wyniku pipeline
// i zarządza flagami + personą

use std::collections::HashSet;
use crate::rules::{
    Flag, Persona, Rule, RuleAction, RuleCondition,
    ResponseTemplate, all_personas, all_rules, all_templates,
};
use crate::neuron::PipelineResult;

// ========================
// STAN BEHAWIORALNY
// ========================

pub struct BehaviorState {
    pub active_flags:    HashSet<String>,
    pub current_persona: String,
    pub personas:        Vec<Persona>,
    pub rules:           Vec<Rule>,
    pub templates:       Vec<ResponseTemplate>,
    pub rng_counter:     usize, // prosty pseudolosowy wybór wariantu
}

impl BehaviorState {
    pub fn new() -> Self {
        let mut state = BehaviorState {
            active_flags:    HashSet::new(),
            current_persona: "default".into(),
            personas:        all_personas(),
            rules:           all_rules(),
            templates:       all_templates(),
            rng_counter:     0,
        };

        // Domyślne flagi startowe
        state.set_flag(Flag::TextMode);
        state.set_flag(Flag::CuriousMode);
        state.set_flag(Flag::CasualMode);

        state
    }

    pub fn set_flag(&mut self, flag: Flag) {
        self.active_flags.insert(flag.as_str().to_string());
    }

    pub fn clear_flag(&mut self, flag: Flag) {
        self.active_flags.remove(flag.as_str());
    }

    pub fn has_flag(&self, flag: &Flag) -> bool {
        self.active_flags.contains(flag.as_str())
    }

    pub fn set_persona(&mut self, id: &str) {
        if self.personas.iter().any(|p| p.id == id) {
            self.current_persona = id.to_string();
        }
    }

    pub fn current_persona(&self) -> &Persona {
        self.personas.iter()
            .find(|p| p.id == self.current_persona)
            .unwrap_or(&self.personas[0])
    }

    /// Prosty deterministyczny wybór wariantu (round-robin)
    fn pick_variant<'a>(&mut self, variants: &'a [&'static str]) -> &'a str {
        if variants.is_empty() { return ""; }
        let idx = self.rng_counter % variants.len();
        self.rng_counter += 1;
        variants[idx]
    }
}

// ========================
// SILNIK BEHAWIORALNY
// ========================

pub struct BehaviorEngine;

impl BehaviorEngine {
    /// Główna funkcja — przetwarza wynik pipeline i zwraca odpowiedź
    pub fn process(
        state:     &mut BehaviorState,
        result:    &PipelineResult,
        input_text: &str,
    ) -> BehaviorResponse {
        let intent  = result.intent();
        let signal  = result.generation.signal;
        let input_lower = input_text.to_lowercase();

        // Sortuj reguły po priorytecie (malejąco)
        let mut sorted_rules = state.rules.clone();
        sorted_rules.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        // Znajdź pasującą regułę
        let matched_rule = sorted_rules.iter().find(|rule| {
            Self::matches_condition(&rule.condition, &intent, signal, &input_lower, state)
        });

        let mut response = BehaviorResponse::default();

        if let Some(rule) = matched_rule {
            let action = rule.action.clone();
            Self::execute_action(action, state, &mut response);
            response.rule_used = rule.id.to_string();
        } else {
            // Fallback jeśli żadna reguła nie pasuje
            response.text     = Self::get_template(state, "fallback", &input_lower);
            response.rule_used = "fallback_default".into();
        }

        // Jeśli tekst pusty po wykonaniu akcji — użyj fallback
        if response.text.is_empty() {
            response.text = Self::get_template(state, "fallback", &input_lower);
        }

        response.persona  = state.current_persona.clone();
        response.flags    = state.active_flags.iter().cloned().collect();
        response
    }

    /// Sprawdza czy warunek reguły jest spełniony
    fn matches_condition(
        cond:        &RuleCondition,
        intent:      &str,
        signal:      f32,
        input_lower: &str,
        state:       &BehaviorState,
    ) -> bool {
        // Sprawdź intencję
        if let Some(req_intent) = cond.intent {
            let intent_short = intent.replace("INTENT:", "");
            if intent_short != req_intent { return false; }
        }

        // Sprawdź minimalny sygnał
        if signal < cond.min_signal { return false; }

        // Sprawdź flags_all — wszystkie muszą być aktywne
        if !cond.flags_all.is_empty() {
            if !cond.flags_all.iter().all(|f| state.has_flag(f)) {
                return false;
            }
        }

        // Sprawdź flags_any — przynajmniej jedna musi być aktywna
        if !cond.flags_any.is_empty() {
            if !cond.flags_any.iter().any(|f| state.has_flag(f)) {
                return false;
            }
        }

        // Sprawdź flags_none — żadna nie może być aktywna
        if !cond.flags_none.is_empty() {
            if cond.flags_none.iter().any(|f| state.has_flag(f)) {
                return false;
            }
        }

        // Sprawdź słowa kluczowe
        if !cond.keywords.is_empty() {
            if !cond.keywords.iter().any(|kw| input_lower.contains(kw)) {
                return false;
            }
        }

        true
    }

    /// Wykonuje akcję reguły
    fn execute_action(
        action:   RuleAction,
        state:    &mut BehaviorState,
        response: &mut BehaviorResponse,
    ) {
        match action {
            RuleAction::UseTemplate(id) => {
                response.text = Self::get_template(state, id, "");
            }
            RuleAction::SetFlag(flag) => {
                state.set_flag(flag);
            }
            RuleAction::ClearFlag(flag) => {
                state.clear_flag(flag);
            }
            RuleAction::SetPersona(id) => {
                state.set_persona(id);
            }
            RuleAction::AskFollowUp(template_id) => {
                response.text        = Self::get_template(state, template_id, "");
                response.is_question = true;
            }
            RuleAction::Multi(actions) => {
                for a in actions {
                    Self::execute_action(a, state, response);
                }
            }
        }
    }

    /// Pobiera tekst z szablonu (z uwzględnieniem persony)
    fn get_template(
        state:       &mut BehaviorState,
        template_id: &str,
        _context:    &str,
    ) -> String {
        let persona_id = state.current_persona.clone();

        // Szukaj szablonu pasującego do persony
        let template = state.templates.iter()
            .find(|t| {
                t.id == template_id
                && t.persona.map(|p| p == persona_id).unwrap_or(true)
            })
            .or_else(|| {
                // Fallback: szablon bez persony
                state.templates.iter().find(|t| t.id == template_id)
            });

        if let Some(tmpl) = template {
            let variants = tmpl.variants;
            if variants.is_empty() {
                return String::new();
            }
            let idx = state.rng_counter % variants.len();
            state.rng_counter += 1;
            variants[idx].to_string()
        } else {
            String::new()
        }
    }

    /// Wykryj flagi z tekstu użytkownika (auto-detect)
    pub fn detect_flags_from_input(
        state:      &mut BehaviorState,
        input_text: &str,
    ) {
        let lower = input_text.to_lowercase();

        // Wykryj techniczny charakter
        if lower.contains("neuron") || lower.contains("token")
        || lower.contains("pipeline") || lower.contains("rust")
        || lower.contains("asm") || lower.contains("kod") {
            state.set_flag(Flag::UserTechnical);
        }

        // Wykryj frustrację
        if lower.contains("nie rozumiesz") || lower.contains("głupia")
        || lower.contains("stupid") || lower.contains("broken")
        || lower.contains("nie działa") {
            state.set_flag(Flag::UserAngry);
        } else {
            state.clear_flag(Flag::UserAngry);
        }

        // Wykryj przyjazny ton
        if lower.contains("dziękuję") || lower.contains("dzięki")
        || lower.contains("thanks") || lower.contains("great")
        || lower.contains("super") || lower.contains("świetnie") {
            state.set_flag(Flag::UserFriendly);
        }

        // Wyczyść WaitingAnswer jeśli użytkownik odpowiedział
        if state.has_flag(&Flag::WaitingAnswer) {
            state.clear_flag(Flag::WaitingAnswer);
        }
    }
}

// ========================
// WYNIK BEHAWIORALNY
// ========================

#[derive(Debug, Default)]
pub struct BehaviorResponse {
    pub text:        String,
    pub rule_used:   String,
    pub persona:     String,
    pub is_question: bool,
    pub flags:       Vec<String>,
}

impl BehaviorResponse {
    pub fn print_debug(&self) {
        println!("[BEHAVIOR] rule={} persona={} question={}",
            self.rule_used, self.persona, self.is_question);
        println!("[BEHAVIOR] flags={:?}", self.flags);
        println!("[BEHAVIOR] response=\"{}\"", &self.text[..self.text.len().min(80)]);
    }
}
