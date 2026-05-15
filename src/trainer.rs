// src/trainer.rs
// System auto-szkolenia Aurory
// Ocenia odpowiedzi, koryguje wagi neuronów na bieżąco
// Zapisuje stan treningu do aurora_training.json

use std::collections::HashMap;
use std::fs;
use crate::neuron::{NeuralPipeline, PipelineResult};
use crate::behavior::BehaviorResponse;

// ========================
// OCENA ODPOWIEDZI
// ========================

#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub input:          String,
    pub intent_detected: String,
    pub rule_used:      String,
    pub response:       String,
    pub score:          f32,      // 0.0-1.0 jak dobra była odpowiedź
    pub corrections:    Vec<Correction>,
}

#[derive(Debug, Clone)]
pub struct Correction {
    pub neuron_id: usize,
    pub field:     CorrectionField,
    pub delta:     f32,  // o ile zmienić wagę
}

#[derive(Debug, Clone)]
pub enum CorrectionField {
    Threshold,
    Bias,
    SynapseWeight(usize), // indeks synapsy
}

// ========================
// KRYTERIA OCENY
// ========================

struct ScoringCriteria;

impl ScoringCriteria {
    /// Ocenia jakość odpowiedzi na podstawie wyniku pipeline i odpowiedzi behawioralnej
    fn score(
        result:   &PipelineResult,
        behavior: &BehaviorResponse,
        input:    &str,
    ) -> f32 {
        let mut score = 0.0_f32;
        let mut weight_sum = 0.0_f32;

        // [1] Czy pipeline w ogóle wystrzelił?
        let fired_count = [
            result.semantic.fired,
            result.grammar.fired,
            result.sentence.fired,
            result.meaning.fired,
            result.process.fired,
            result.generation.fired,
        ].iter().filter(|&&f| f).count();

        let fire_ratio = fired_count as f32 / 6.0;
        score      += fire_ratio * 0.25;
        weight_sum += 0.25;

        // [2] Czy intencja jest znana?
        let intent_known = !behavior.rule_used.contains("fallback");
        score      += if intent_known { 0.3 } else { 0.0 };
        weight_sum += 0.3;

        // [3] Jakość sygnału generacji
        score      += result.generation.signal * 0.2;
        weight_sum += 0.2;

        // [4] Czy odpowiedź nie jest pusta?
        let response_ok = !behavior.text.is_empty() && behavior.text.len() > 10;
        score      += if response_ok { 0.15 } else { 0.0 };
        weight_sum += 0.15;

        // [5] Spójność intencji z regułą
        let coherent = Self::check_coherence(result, behavior);
        score      += if coherent { 0.1 } else { 0.0 };
        weight_sum += 0.1;

        if weight_sum > 0.0 { score / weight_sum } else { 0.0 }
    }

    fn check_coherence(result: &PipelineResult, behavior: &BehaviorResponse) -> bool {
        let intent = result.intent();
        let rule   = &behavior.rule_used;

        // Reguły muszą pasować do intencji
        match intent.as_str() {
            "INTENT:QUERY_GENERAL" | "INTENT:QUERY_ENTITY" => {
                rule.contains("identity") || rule.contains("tech") || rule.contains("greet")
            }
            "INTENT:EXECUTE_COMMAND" => {
                rule.contains("cmd") || rule.contains("command")
            }
            "INTENT:STATEMENT_FACT" | "INTENT:STATEMENT_GENERAL" => {
                rule.contains("statement") || rule.contains("curious") || rule.contains("fallback")
            }
            _ => true,
        }
    }
}

// ========================
// WAGI TRENINGOWE
// ========================

#[derive(Debug, Clone)]
pub struct TrainingWeights {
    /// Korekty progów neuronów: neuron_id -> threshold
    pub thresholds:     HashMap<usize, f32>,
    /// Korekty biasów: neuron_id -> bias
    pub biases:         HashMap<usize, f32>,
    /// Korekty wag synaps: (neuron_id, synapse_idx) -> weight
    pub synapse_weights: HashMap<(usize, usize), f32>,
    /// Historia scorów (ostatnie N)
    pub score_history:  Vec<f32>,
    /// Łączna liczba iteracji treningu
    pub iterations:     usize,
    /// Średni score
    pub avg_score:      f32,
}

impl Default for TrainingWeights {
    fn default() -> Self {
        TrainingWeights {
            thresholds:      HashMap::new(),
            biases:          HashMap::new(),
            synapse_weights: HashMap::new(),
            score_history:   Vec::new(),
            iterations:      0,
            avg_score:       0.0,
        }
    }
}

impl TrainingWeights {
    pub fn save(&self, path: &str) {
        // Serializacja ręczna do JSON (bez serde żeby nie dodawać zależności)
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"iterations\": {},\n", self.iterations));
        json.push_str(&format!("  \"avg_score\": {},\n", self.avg_score));

        json.push_str("  \"thresholds\": {");
        let thresh_entries: Vec<String> = self.thresholds.iter()
            .map(|(k, v)| format!("\"{}\": {:.4}", k, v))
            .collect();
        json.push_str(&thresh_entries.join(", "));
        json.push_str("},\n");

        json.push_str("  \"biases\": {");
        let bias_entries: Vec<String> = self.biases.iter()
            .map(|(k, v)| format!("\"{}\": {:.4}", k, v))
            .collect();
        json.push_str(&bias_entries.join(", "));
        json.push_str("},\n");

        let recent: Vec<String> = self.score_history.iter()
            .rev().take(20)
            .map(|s| format!("{:.3}", s))
            .collect();
        json.push_str(&format!("  \"recent_scores\": [{}]\n", recent.join(", ")));
        json.push_str("}\n");

        match fs::write(path, &json) {
            Ok(_)  => println!("[TRAINER] Wagi zapisane -> {}", path),
            Err(e) => eprintln!("[TRAINER] Błąd zapisu: {}", e),
        }
    }

    pub fn load(&mut self, path: &str) {
        match fs::read_to_string(path) {
            Ok(json) => {
                // Prosta ekstrakcja liczb z JSON bez parsera
                if let Some(iter) = Self::extract_f32(&json, "iterations") {
                    self.iterations = iter as usize;
                }
                if let Some(avg) = Self::extract_f32(&json, "avg_score") {
                    self.avg_score = avg;
                }
                println!("[TRAINER] Wagi wczytane z {} (iter={})", path, self.iterations);
            }
            Err(_) => {
                println!("[TRAINER] Brak pliku wag — zaczynam od zera");
            }
        }
    }

    fn extract_f32(json: &str, key: &str) -> Option<f32> {
        let pattern = format!("\"{}\":", key);
        let pos = json.find(&pattern)?;
        let after = &json[pos + pattern.len()..];
        let trimmed = after.trim_start();
        let end = trimmed.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')?;
        trimmed[..end].parse().ok()
    }
}

// ========================
// TRAINER
// ========================

pub struct Trainer {
    pub weights:  TrainingWeights,
    pub history:  Vec<TrainingRecord>,
    pub save_path: String,
    // Parametry uczenia
    learning_rate:  f32,
    history_limit:  usize,
    save_every:     usize, // zapisuj co N iteracji
}

impl Trainer {
    pub fn new(save_path: &str) -> Self {
        let mut trainer = Trainer {
            weights:       TrainingWeights::default(),
            history:       Vec::new(),
            save_path:     save_path.to_string(),
            learning_rate: 0.05,
            history_limit: 200,
            save_every:    10,
        };
        trainer.weights.load(save_path);
        trainer
    }

    /// Główna funkcja treningu — ocenia i koryguje po każdej odpowiedzi
    pub fn train_step(
        &mut self,
        pipeline: &mut NeuralPipeline,
        result:   &PipelineResult,
        behavior: &BehaviorResponse,
        input:    &str,
    ) -> f32 {
        // Oceń odpowiedź
        let score = ScoringCriteria::score(result, behavior, input);

        // Oblicz korekty
        let corrections = self.compute_corrections(result, score);

        // Zastosuj korekty do neuronów
        self.apply_corrections(pipeline, &corrections);

        // Zapisz rekord
        let record = TrainingRecord {
            input:           input.to_string(),
            intent_detected: result.intent(),
            rule_used:       behavior.rule_used.clone(),
            response:        behavior.text.clone(),
            score,
            corrections:     corrections.clone(),
        };

        self.history.push(record);
        if self.history.len() > self.history_limit {
            self.history.remove(0);
        }

        // Aktualizuj statystyki
        self.weights.score_history.push(score);
        if self.weights.score_history.len() > 100 {
            self.weights.score_history.remove(0);
        }
        self.weights.iterations += 1;
        self.weights.avg_score = self.weights.score_history.iter().sum::<f32>()
            / self.weights.score_history.len() as f32;

        // Zapisuj co N iteracji
        if self.weights.iterations % self.save_every == 0 {
            self.weights.save(&self.save_path);
        }

        println!("[TRAINER] iter={} score={:.3} avg={:.3}",
            self.weights.iterations, score, self.weights.avg_score);

        score
    }

    /// Oblicza korekty wag na podstawie score
    fn compute_corrections(
        &self,
        result: &PipelineResult,
        score:  f32,
    ) -> Vec<Correction> {
        let mut corrections = Vec::new();
        let lr = self.learning_rate;

        // Jeśli score niski — zmniejsz progi neuronów które nie wystrzeliły
        let neuron_signals = [
            (0, result.semantic.signal,   result.semantic.fired),
            (1, result.grammar.signal,    result.grammar.fired),
            (2, result.command.signal,    result.command.fired),
            (3, result.sentence.signal,   result.sentence.fired),
            (4, result.meaning.signal,    result.meaning.fired),
            (5, result.process.signal,    result.process.fired),
            (6, result.generation.signal, result.generation.fired),
        ];

        for (neuron_id, signal, fired) in &neuron_signals {
            if score < 0.4 {
                // Słaba odpowiedź — obniż próg żeby neuron łatwiej strzelał
                if !fired && *signal > 0.1 {
                    corrections.push(Correction {
                        neuron_id: *neuron_id,
                        field:     CorrectionField::Threshold,
                        delta:     -lr * (1.0 - score),
                    });
                }
                // Zwiększ bias dla neuronów ze słabym sygnałem
                if *signal < 0.3 {
                    corrections.push(Correction {
                        neuron_id: *neuron_id,
                        field:     CorrectionField::Bias,
                        delta:     lr * 0.5,
                    });
                }
            } else if score > 0.7 {
                // Dobra odpowiedź — wzmocnij neurony które wystrzeliły
                if *fired {
                    corrections.push(Correction {
                        neuron_id: *neuron_id,
                        field:     CorrectionField::Bias,
                        delta:     lr * 0.2 * score,
                    });
                }
            }

            // Korekta wag synaps
            if *fired && score < 0.5 {
                corrections.push(Correction {
                    neuron_id: *neuron_id,
                    field:     CorrectionField::SynapseWeight(0),
                    delta:     lr * (score - 0.5),
                });
            }
        }

        corrections
    }

    /// Aplikuje korekty bezpośrednio do neuronów w pipeline
    fn apply_corrections(
        &mut self,
        pipeline:    &mut NeuralPipeline,
        corrections: &[Correction],
    ) {
        for correction in corrections {
            let nid = correction.neuron_id;
            if nid >= pipeline.neurons.len() { continue; }

            let neuron = &mut pipeline.neurons[nid];
            let delta  = correction.delta;

            match &correction.field {
                CorrectionField::Threshold => {
                    neuron.threshold = (neuron.threshold + delta).clamp(0.05, 0.95);
                    self.weights.thresholds.insert(nid, neuron.threshold);
                }
                CorrectionField::Bias => {
                    neuron.bias = (neuron.bias + delta).clamp(-1.0, 1.0);
                    self.weights.biases.insert(nid, neuron.bias);
                }
                CorrectionField::SynapseWeight(syn_idx) => {
                    if *syn_idx < neuron.synapses.len() {
                        let new_w = (neuron.synapses[*syn_idx].weight + delta).clamp(0.0, 2.0);
                        neuron.synapses[*syn_idx].weight = new_w;
                        self.weights.synapse_weights.insert((nid, *syn_idx), new_w);
                    }
                }
            }
        }
    }

    /// Zastosuj zapisane wagi do pipeline (po wczytaniu)
    pub fn apply_saved_weights(&self, pipeline: &mut NeuralPipeline) {
        for (&nid, &threshold) in &self.weights.thresholds {
            if nid < pipeline.neurons.len() {
                pipeline.neurons[nid].threshold = threshold;
            }
        }
        for (&nid, &bias) in &self.weights.biases {
            if nid < pipeline.neurons.len() {
                pipeline.neurons[nid].bias = bias;
            }
        }
        for (&(nid, syn_idx), &weight) in &self.weights.synapse_weights {
            if nid < pipeline.neurons.len()
            && syn_idx < pipeline.neurons[nid].synapses.len() {
                pipeline.neurons[nid].synapses[syn_idx].weight = weight;
            }
        }
        if self.weights.iterations > 0 {
            println!("[TRAINER] Zastosowano wagi z {} iteracji (avg_score={:.3})",
                self.weights.iterations, self.weights.avg_score);
        }
    }

    /// Podsumowanie treningu
    pub fn summary(&self) -> String {
        let last_scores: Vec<f32> = self.weights.score_history
            .iter().rev().take(10).cloned().collect();

        let trend = if last_scores.len() >= 2 {
            let first = last_scores.last().unwrap_or(&0.0);
            let last  = last_scores.first().unwrap_or(&0.0);
            if last > first { "↑ poprawia się" }
            else if last < first { "↓ pogarsza się" }
            else { "→ stabilny" }
        } else {
            "— za mało danych"
        };

        format!(
            "Iteracje: {} | Avg score: {:.3} | Trend: {} | Historia: {} rekordów",
            self.weights.iterations,
            self.weights.avg_score,
            trend,
            self.history.len()
        )
    }
}
