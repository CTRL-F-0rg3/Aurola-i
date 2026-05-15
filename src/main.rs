// src/main.rs
// Aurora — łączy tokenizer + pipeline neuronów + GUI chat

mod tokenizer;
mod neuron;
mod rules;
mod behavior;
mod trainer;

use eframe::egui;
use eframe::egui::{
    Color32, FontId, Frame, Margin, Rounding, Stroke, Vec2,
    FontFamily, RichText, ScrollArea, TextEdit, Key,
};

use tokenizer::{VocabTables, Tokenizer};
use neuron::{NeuralPipeline, PipelineResult, TokenContext};
use behavior::{BehaviorEngine, BehaviorState};
use trainer::Trainer;

// ========================
// PALETA (ta sama co GUI)
// ========================

const BG_DARKEST:    Color32 = Color32::from_rgb(10,  6,  20);
const BG_DARK:       Color32 = Color32::from_rgb(18, 12, 35);
const BG_PANEL:      Color32 = Color32::from_rgb(25, 16, 50);
const BG_WIDGET:     Color32 = Color32::from_rgb(35, 22, 68);
const PURPLE_LIGHT:  Color32 = Color32::from_rgb(180, 130, 255);
const PURPLE_MID:    Color32 = Color32::from_rgb(140,  80, 230);
const PURPLE_ACCENT: Color32 = Color32::from_rgb(200, 100, 255);
const PURPLE_DIM:    Color32 = Color32::from_rgb( 90,  50, 150);
const TEXT_PRIMARY:  Color32 = Color32::from_rgb(220, 200, 255);
const TEXT_DIM:      Color32 = Color32::from_rgb(130, 100, 180);
const TEXT_BRIGHT:   Color32 = Color32::from_rgb(255, 240, 255);
const GREEN_OK:      Color32 = Color32::from_rgb( 80, 220, 140);
const ORANGE_WARN:   Color32 = Color32::from_rgb(255, 160,  60);

// ========================
// WIADOMOŚĆ W CHACIE
// ========================

#[derive(Debug, Clone, PartialEq)]
enum Speaker { User, Aurora }

#[derive(Debug, Clone)]
struct ChatMessage {
    speaker: Speaker,
    text:    String,
    // Debug info z pipeline
    intent:  Option<String>,
    tokens:  Option<Vec<String>>,
    signals: Option<[f32; 7]>,
}

impl ChatMessage {
    fn user(text: String) -> Self {
        ChatMessage { speaker: Speaker::User, text, intent: None, tokens: None, signals: None }
    }

    fn aurora(text: String, intent: String, tokens: Vec<String>, signals: [f32; 7]) -> Self {
        ChatMessage {
            speaker: Speaker::Aurora,
            text,
            intent:  Some(intent),
            tokens:  Some(tokens),
            signals: Some(signals),
        }
    }
}

// ========================
// STAN APLIKACJI
// ========================

struct AuroraApp {
    tokenizer:   Tokenizer,
    pipeline:    NeuralPipeline,
    behavior:    BehaviorState,
    trainer:     Trainer,
    vocab_rev:   std::collections::HashMap<u32, String>, // id -> słowo (do generacji)

    messages:    Vec<ChatMessage>,
    input:       String,
    show_debug:  bool,
    status:      String,
    name:        String,  // nazwa AI
}

impl AuroraApp {
    fn new() -> Self {
        println!("[AURORA] Ładowanie vocab...");
        let tables = VocabTables::load(
            "src/vocab.asm",
            "src/vocab_pos.asm",
            "src/vocab_lemma.asm",
        );

        // Zbuduj odwrócony słownik id -> słowo
        let mut vocab_rev = std::collections::HashMap::new();
        for (word, id) in &tables.vocab {
            vocab_rev.insert(*id, word.clone());
        }

        let tokenizer = Tokenizer::new(tables);
        let mut pipeline = NeuralPipeline::new();
        let behavior  = BehaviorState::new();
        let trainer   = Trainer::new("aurora_training.json");

        // Zastosuj zapisane wagi treningu
        trainer.apply_saved_weights(&mut pipeline);

        let mut app = AuroraApp {
            tokenizer,
            pipeline,
            behavior,
            trainer,
            vocab_rev,
            messages:   Vec::new(),
            input:      String::new(),
            show_debug: false,
            status:     "Gotowa".into(),
            name:       "Aurora".into(),
        };

        // Wiadomość powitalna
        app.messages.push(ChatMessage::aurora(
            "Cześć. Jestem Aurora. Mówię do ciebie przez pipeline neuronowy — każde twoje słowo przechodzi przez 7 warstw zanim odpowiem.".into(),
            "INTENT:GREETING".into(),
            vec![],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ));

        app
    }

    /// Główna funkcja przetwarzania wejścia
    fn process_input(&mut self, text: &str) {
        if text.trim().is_empty() { return; }

        // Dodaj wiadomość użytkownika
        self.messages.push(ChatMessage::user(text.to_string()));

        // Tokenizuj
        let tokens = self.tokenizer.tokenize(text);

        // Konwertuj Token -> TokenContext (którego używa pipeline)
        let token_contexts: Vec<TokenContext> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| TokenContext {
                id:       t.id,
                word:     t.word.clone(),
                pos_id:   t.pos_id,
                lemma_id: t.lemma_id,
                position: i,
            })
            .collect();

        // Przepuść przez pipeline
        let result = self.pipeline.run(&token_contexts);

        // Wykryj flagi z tekstu
        BehaviorEngine::detect_flags_from_input(&mut self.behavior, text);

        // Silnik behawioralny — wybiera odpowiedź na podstawie reguł i flag
        let behavior_resp = BehaviorEngine::process(
            &mut self.behavior,
            &result,
            text,
        );
        behavior_resp.print_debug();

        // Auto-szkolenie — oceń i skoryguj wagi
        let score = self.trainer.train_step(
            &mut self.pipeline,
            &result,
            &behavior_resp,
            text,
        );

        // Zbierz sygnały z każdej warstwy
        let signals = [
            result.semantic.signal,
            result.grammar.signal,
            result.command.signal,
            result.sentence.signal,
            result.meaning.signal,
            result.process.signal,
            result.generation.signal,
        ];

        // Zbierz tagi tokenów odpowiedzi
        let response_token_words: Vec<String> = result.generation.tokens
            .iter()
            .filter_map(|id| self.vocab_rev.get(id))
            .cloned()
            .collect();

        let intent = result.intent();
        let response_text = behavior_resp.text;

        self.messages.push(ChatMessage::aurora(
            response_text,
            intent,
            response_token_words,
            signals,
        ));

        result.print_summary();
        self.status = format!("Przetworzono | score={:.2} | {}", score, self.trainer.summary());
    }

    /// Generuje odpowiedź tekstową na podstawie wyniku pipeline
    fn generate_response(
        &self,
        result: &PipelineResult,
        input_tokens: &[TokenContext],
    ) -> String {
        let intent = result.intent();

        match intent.as_str() {

            "INTENT:QUERY_ENTITY" | "INTENT:QUERY_GENERAL" => {
                // Pytanie — spróbuj odpowiedzieć
                let subject = result.sentence.tags.iter()
                    .find(|t| t.starts_with("SUBJECT:"))
                    .map(|s| s.replace("SUBJECT:", ""))
                    .unwrap_or_default();

                // Pytanie o Aurorę
                if subject.to_uppercase() == "AURORA"
                || input_tokens.iter().any(|t| t.word.to_uppercase() == "AURORA") {
                    format!(
                        "Jestem {}. Przetwarzam tekst przez {} warstw neuronowych. \
                         Aktualnie rozumiem {} tokenów w słowniku.",
                        self.name,
                        7,
                        self.vocab_rev.len()
                    )
                } else if subject.to_uppercase() == "YOU"
                       || subject.to_uppercase() == "YOUR" {
                    format!(
                        "Jestem modelem językowym o nazwie {}. \
                         Przetwarzam zdania przez pipeline: \
                         Semantic → Grammar → Command → Sentence → Meaning → Process → Generation.",
                        self.name
                    )
                } else {
                    // Ogólne pytanie
                    let key_words: Vec<String> = input_tokens.iter()
                        .filter(|t| matches!(t.pos_id, 1 | 2 | 10))
                        .map(|t| t.word.clone())
                        .collect();

                    if key_words.is_empty() {
                        "Rozumiem twoje pytanie, ale nie mam jeszcze wystarczającej wiedzy by odpowiedzieć dokładnie.".into()
                    } else {
                        format!(
                            "Pytasz o: {}. Mój pipeline wykrył intencję: {}. \
                             Potrzebuję więcej wiedzy by odpowiedzieć szczegółowo.",
                            key_words.join(", "),
                            intent.replace("INTENT:", "")
                        )
                    }
                }
            }

            "INTENT:EXECUTE_COMMAND" | "INTENT:REQUEST_ACTION" => {
                // Polecenie
                let cmd_tags: Vec<String> = result.command.tags.iter()
                    .filter(|t| t.starts_with("CMD:"))
                    .map(|s| s.replace("CMD:", ""))
                    .collect();

                if cmd_tags.is_empty() {
                    "Rozumiem że chcesz żebym coś zrobiła. Jakie dokładnie polecenie?".into()
                } else {
                    format!(
                        "Wykryłam polecenie: {}. \
                         Wykonanie poleceń będzie dostępne gdy zostanie zaimplementowany moduł akcji.",
                        cmd_tags.join(", ")
                    )
                }
            }

            "INTENT:STATEMENT_FACT" | "INTENT:STATEMENT_GENERAL" => {
                // Stwierdzenie
                let svo = result.sentence.data.get("svo_score").copied().unwrap_or(0.0);
                let predicate = result.sentence.tags.iter()
                    .find(|t| t.starts_with("PREDICATE:"))
                    .map(|s| s.replace("PREDICATE:", ""))
                    .unwrap_or_default();

                if svo > 0.8 {
                    format!(
                        "Rozumiem. Akcja \"{}\" została zarejestrowana z pewnością {:.0}%.",
                        predicate,
                        svo * 100.0
                    )
                } else {
                    "Rozumiem twoje zdanie. Struktura gramatyczna jest częściowa.".into()
                }
            }

            _ => {
                // Fallback
                let sem_tags: Vec<&String> = result.semantic.tags.iter().take(3).collect();
                format!(
                    "Przetworzyłam {} tokenów. Wykryte kategorie: {}. \
                     Pewność przetwarzania: {:.0}%.",
                    input_tokens.len(),
                    sem_tags.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                    result.generation.signal * 100.0
                )
            }
        }
    }
}

// ========================
// STYL
// ========================

fn apply_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals.window_fill      = BG_DARK;
    style.visuals.panel_fill       = BG_DARKEST;
    style.visuals.extreme_bg_color = BG_DARKEST;
    style.visuals.widgets.inactive.bg_fill   = BG_WIDGET;
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, PURPLE_DIM);
    style.visuals.widgets.inactive.rounding  = Rounding::same(6.0);
    style.visuals.widgets.hovered.bg_fill    = Color32::from_rgb(50, 32, 95);
    style.visuals.widgets.hovered.fg_stroke  = Stroke::new(1.5, PURPLE_LIGHT);
    style.visuals.widgets.active.bg_fill     = PURPLE_MID;
    style.visuals.selection.bg_fill          = PURPLE_MID;
    style.visuals.window_stroke  = Stroke::new(1.0, PURPLE_DIM);
    style.spacing.item_spacing   = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);
    ctx.set_style(style);
}

// ========================
// EGUI APP
// ========================

impl eframe::App for AuroraApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_style(ctx);

        // Topbar
        egui::TopBottomPanel::top("top")
            .frame(Frame::none().fill(BG_DARKEST).inner_margin(Margin::symmetric(16.0, 10.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("◈ AURORA")
                            .color(PURPLE_ACCENT)
                            .font(FontId::new(20.0, FontFamily::Monospace))
                            .strong()
                    );
                    ui.label(
                        RichText::new("  Neural Chat")
                            .color(TEXT_DIM)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(
                            egui::Button::new(
                                RichText::new(if self.show_debug { "[ debug ON ]" } else { "[ debug OFF ]" })
                                    .color(if self.show_debug { ORANGE_WARN } else { TEXT_DIM })
                                    .font(FontId::new(11.0, FontFamily::Monospace))
                            )
                            .fill(Color32::TRANSPARENT)
                        ).clicked() {
                            self.show_debug = !self.show_debug;
                        }
                        ui.label(
                            RichText::new(&self.status)
                                .color(GREEN_OK)
                                .font(FontId::new(11.0, FontFamily::Monospace))
                        );
                    });
                });
            });

        // Input na dole
        let mut send = false;
        egui::TopBottomPanel::bottom("input_panel")
            .frame(Frame::none().fill(BG_DARK).inner_margin(Margin::same(12.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let input_field = ui.add(
                        TextEdit::singleline(&mut self.input)
                            .font(FontId::new(13.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY)
                            .hint_text("Napisz do Aurory...")
                            .desired_width(ui.available_width() - 90.0)
                            .frame(true)
                    );

                    // Enter = wyślij
                    if input_field.lost_focus() && ctx.input(|i| i.key_pressed(Key::Enter)) {
                        send = true;
                        input_field.request_focus();
                    }

                    if ui.add(
                        egui::Button::new(
                            RichText::new("Wyślij")
                                .color(TEXT_BRIGHT)
                                .font(FontId::new(12.0, FontFamily::Monospace))
                        )
                        .fill(PURPLE_MID)
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(80.0, 32.0))
                    ).clicked() {
                        send = true;
                    }
                });
            });

        // Wyślij wiadomość
        if send && !self.input.trim().is_empty() {
            let text = self.input.clone();
            self.input.clear();
            self.process_input(&text);
        }

        // Główny panel — historia czatu
        egui::CentralPanel::default()
            .frame(Frame::none().fill(BG_DARKEST).inner_margin(Margin::symmetric(16.0, 12.0)))
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for msg in &self.messages {
                            self.render_message(ui, msg);
                            ui.add_space(8.0);
                        }
                    });
            });
    }
}

impl AuroraApp {
    fn render_message(&self, ui: &mut egui::Ui, msg: &ChatMessage) {
        let is_aurora = msg.speaker == Speaker::Aurora;

        let (bg, label_color, label) = if is_aurora {
            (BG_PANEL, PURPLE_ACCENT, "Aurora")
        } else {
            (BG_WIDGET, GREEN_OK, "Ty")
        };

        Frame {
            fill:         bg,
            rounding:     Rounding::same(8.0),
            stroke:       Stroke::new(1.0, if is_aurora { PURPLE_DIM } else { Color32::from_rgb(40, 80, 40) }),
            inner_margin: Margin::same(10.0),
            outer_margin: Margin::symmetric(if is_aurora { 0.0 } else { 60.0 }, 0.0),
            ..Default::default()
        }.show(ui, |ui| {
            // Nagłówek
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(label)
                        .color(label_color)
                        .font(FontId::new(11.0, FontFamily::Monospace))
                        .strong()
                );
                if is_aurora {
                    if let Some(intent) = &msg.intent {
                        ui.label(
                            RichText::new(format!("  {}", intent.replace("INTENT:", "")))
                                .color(TEXT_DIM)
                                .font(FontId::new(10.0, FontFamily::Monospace))
                        );
                    }
                }
            });

            ui.add_space(4.0);

            // Treść wiadomości
            ui.label(
                RichText::new(&msg.text)
                    .color(TEXT_PRIMARY)
                    .font(FontId::new(13.0, FontFamily::Proportional))
            );

            // Debug info (tylko Aurora, tylko gdy show_debug)
            if is_aurora && self.show_debug {
                if let Some(signals) = &msg.signals {
                    ui.add_space(6.0);
                    ui.separator();

                    let layer_names = [
                        "Semantic", "Grammar", "Command",
                        "Sentence", "Meaning", "Process", "Generation"
                    ];
                    let colors = [
                        PURPLE_LIGHT, PURPLE_LIGHT, ORANGE_WARN,
                        PURPLE_LIGHT, PURPLE_ACCENT, PURPLE_MID, GREEN_OK,
                    ];

                    ui.horizontal_wrapped(|ui| {
                        for (i, (name, &sig)) in layer_names.iter().zip(signals.iter()).enumerate() {
                            let col = if sig > 0.5 { colors[i] } else { TEXT_DIM };
                            ui.label(
                                RichText::new(format!("{}:{:.2}", name, sig))
                                    .color(col)
                                    .font(FontId::new(10.0, FontFamily::Monospace))
                            );
                            if i < 6 { ui.label(RichText::new("→").color(PURPLE_DIM)); }
                        }
                    });

                    if let Some(tokens) = &msg.tokens {
                        if !tokens.is_empty() {
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new(format!("tokens: [{}]", tokens.join(", ")))
                                    .color(TEXT_DIM)
                                    .font(FontId::new(10.0, FontFamily::Monospace))
                            );
                        }
                    }
                }
            }
        });
    }
}

// ========================
// MAIN
// ========================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Aurora — Neural Chat")
            .with_inner_size([900.0, 650.0])
            .with_min_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Aurora",
        options,
        Box::new(|_cc| Box::new(AuroraApp::new())),
    )
}