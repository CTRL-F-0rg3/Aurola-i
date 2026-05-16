// src/behavior.rs (updated)
// Silnik behawioralny z dynamicznymi odpowiedziami opartymi na przykładach

use std::collections::HashSet;
use crate::rules::{
    Flag, Persona, Rule, RuleAction, RuleCondition,
    ResponseTemplate, all_personas, all_rules, all_templates,
};
use crate::neuron::PipelineResult;
use crate::training_data::{TrainingManager, TrainingEval};

pub struct BehaviorState {
    pub active_flags:    HashSet<String>,
    pub current_persona: String,
    pub personas:        Vec<Persona>,
    pub rules:           Vec<Rule>,
    pub templates:       Vec<ResponseTemplate>,
    pub training:        TrainingManager,
    pub rng_counter:     usize,
}

impl BehaviorState {
    pub fn new() -> Self {
        let mut state = BehaviorState {
            active_flags:    HashSet::new(),
            current_persona: "default".into(),
            personas:        all_personas(),
            rules:           all_rules(),
            templates:       all_templates(),
            training:        TrainingManager::new(),
            rng_counter:     0,
        };
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
}

pub struct BehaviorEngine;

impl BehaviorEngine {
    pub fn process(
        state:      &mut BehaviorState,
        result:     &PipelineResult,
        input_text: &str,
    ) -> BehaviorResponse {
        let intent      = result.intent();
        let signal      = result.generation.signal;
        let input_lower = input_text.to_lowercase();

        let mut sorted_rules = state.rules.clone();
        sorted_rules.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        let matched_rule = sorted_rules.iter().find(|rule| {
            Self::matches_condition(&rule.condition, &intent, signal, &input_lower, state)
        });

        let mut response = BehaviorResponse::default();

        if let Some(rule) = matched_rule {
            let action = rule.action.clone();
            response.rule_used = rule.id.to_string();
            Self::execute_action(action, state, &mut response);
        }

        // Jeśli pusta lub fallback — użyj przykładów treningowych
        let is_fallback = response.text.is_empty()
            || response.rule_used.contains("fallback");

        if is_fallback {
            if let Some(ex) = state.training.generate_from_examples(input_text) {
                response.text     = ex;
                response.rule_used = format!("{}_example", response.rule_used);
            }
        }

        if response.text.is_empty() {
            response.text = Self::get_template(state, "fallback", &input_lower);
            if response.rule_used.is_empty() {
                response.rule_used = "fallback_default".into();
            }
        }

        // Ewaluacja przez training manager
        let eval = state.training.evaluate(input_text, &intent, &response.text);
        eval.print_debug();

        // Jeśli training ma lepszą sugestię
        if eval.score < 0.4 {
            if let Some(suggestion) = &eval.suggestion {
                response.text     = suggestion.clone();
                response.rule_used = format!("{}_corrected", response.rule_used);
            }
        }

        response.persona  = state.current_persona.clone();
        response.flags    = state.active_flags.iter().cloned().collect();
        response.eval     = Some(eval);
        response
    }

    fn matches_condition(
        cond:        &RuleCondition,
        intent:      &str,
        signal:      f32,
        input_lower: &str,
        state:       &BehaviorState,
    ) -> bool {
        if let Some(req_intent) = cond.intent {
            let intent_short = intent.replace("INTENT:", "");
            if intent_short != req_intent { return false; }
        }
        if signal < cond.min_signal { return false; }
        if !cond.flags_all.is_empty() {
            if !cond.flags_all.iter().all(|f| state.has_flag(f)) { return false; }
        }
        if !cond.flags_any.is_empty() {
            if !cond.flags_any.iter().any(|f| state.has_flag(f)) { return false; }
        }
        if !cond.flags_none.is_empty() {
            if cond.flags_none.iter().any(|f| state.has_flag(f)) { return false; }
        }
        if !cond.keywords.is_empty() {
            if !cond.keywords.iter().any(|kw| input_lower.contains(kw)) { return false; }
        }
        true
    }

    fn execute_action(
        action:   RuleAction,
        state:    &mut BehaviorState,
        response: &mut BehaviorResponse,
    ) {
        match action {
            RuleAction::UseTemplate(id) => {
                response.text = Self::get_template(state, id, "");
            }
            RuleAction::SetFlag(flag) => { state.set_flag(flag); }
            RuleAction::ClearFlag(flag) => { state.clear_flag(flag); }
            RuleAction::SetPersona(id) => { state.set_persona(id); }
            RuleAction::AskFollowUp(tid) => {
                response.text        = Self::get_template(state, tid, "");
                response.is_question = true;
            }
            RuleAction::Multi(actions) => {
                for a in actions { Self::execute_action(a, state, response); }
            }
        }
    }

    fn get_template(state: &mut BehaviorState, template_id: &str, _ctx: &str) -> String {
        let pid = state.current_persona.clone();
        let tmpl = state.templates.iter()
            .find(|t| t.id == template_id && t.persona.map(|p| p == pid).unwrap_or(true))
            .or_else(|| state.templates.iter().find(|t| t.id == template_id));
        if let Some(t) = tmpl {
            if t.variants.is_empty() { return String::new(); }
            let idx = state.rng_counter % t.variants.len();
            state.rng_counter += 1;
            t.variants[idx].to_string()
        } else {
            String::new()
        }
    }

    pub fn detect_flags_from_input(state: &mut BehaviorState, input_text: &str) {
        let lower = input_text.to_lowercase();
        if lower.contains("neuron") || lower.contains("token")
        || lower.contains("pipeline") || lower.contains("rust")
        || lower.contains("asm") || lower.contains("code") {
            state.set_flag(Flag::UserTechnical);
        }
        if lower.contains("don't understand") || lower.contains("stupid")
        || lower.contains("broken") || lower.contains("not working") {
            state.set_flag(Flag::UserAngry);
        } else { state.clear_flag(Flag::UserAngry); }
        if lower.contains("thank") || lower.contains("great")
        || lower.contains("awesome") { state.set_flag(Flag::UserFriendly); }
        if state.has_flag(&Flag::WaitingAnswer) { state.clear_flag(Flag::WaitingAnswer); }
    }
}

#[derive(Debug, Default)]
pub struct BehaviorResponse {
    pub text:        String,
    pub rule_used:   String,
    pub persona:     String,
    pub is_question: bool,
    pub flags:       Vec<String>,
    pub eval:        Option<TrainingEval>,
}

impl BehaviorResponse {
    pub fn print_debug(&self) {
        println!("[BEHAVIOR] rule={} persona={}", self.rule_used, self.persona);
        if let Some(eval) = &self.eval {
            println!("[BEHAVIOR] eval_score={:.3}", eval.score);
        }
    }

    pub fn eval_score(&self) -> f32 {
        self.eval.as_ref().map(|e| e.score).unwrap_or(0.0)
    }
}