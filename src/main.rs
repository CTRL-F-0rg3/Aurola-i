// src/main.rs
// Aurora — łączy tokenizer + pipeline neuronów + GUI chat

mod tokenizer;
mod neuron;
mod rules;
mod behavior;
mod trainer;
mod training_data;

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
// WIDOK
// ========================

#[derive(PartialEq)]
enum AppView { Chat, Training }

// ========================
// WIADOMOŚĆ W CHACIE
// ========================

#[derive(Debug, Clone, PartialEq)]
enum Speaker { User, Aurora }

#[derive(Debug, Clone)]
struct ChatMessage {
    speaker: Speaker,
    text:    String,
    intent:  Option<String>,
    tokens:  Option<Vec<String>>,
    signals: Option<[f32; 7]>,
    eval_score: Option<f32>,
}

impl ChatMessage {
    fn user(text: String) -> Self {
        ChatMessage { speaker: Speaker::User, text, intent: None, tokens: None, signals: None, eval_score: None }
    }

    fn aurora(text: String, intent: String, tokens: Vec<String>, signals: [f32; 7], eval_score: f32) -> Self {
        ChatMessage {
            speaker: Speaker::Aurora,
            text,
            intent:  Some(intent),
            tokens:  Some(tokens),
            signals: Some(signals),
            eval_score: Some(eval_score),
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
    vocab_rev:   std::collections::HashMap<u32, String>,

    messages:    Vec<ChatMessage>,
    input:       String,
    show_debug:  bool,
    status:      String,
    name:        String,

    // Tab szkolenia
    view:              AppView,
    train_input:       String,
    train_intent:      String,
    train_output:      String,
    train_concepts:    String,
    correction_input:  String,
    correction_bad:    String,
    correction_good:   String,
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
            view:              AppView::Chat,
            train_input:       String::new(),
            train_intent:      "QUERY_GENERAL".into(),
            train_output:      String::new(),
            train_concepts:    String::new(),
            correction_input:  String::new(),
            correction_bad:    String::new(),
            correction_good:   String::new(),
        };

        // Wiadomość powitalna
        app.messages.push(ChatMessage::aurora(
            "Hello. I am Aurora. Every word you type passes through 7 neuron layers before I respond.".into(),
            "INTENT:GREETING".into(),
            vec![],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            1.0,
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

        let intent        = result.intent();
        let eval_score    = behavior_resp.eval_score();
        let response_text = behavior_resp.text;

        self.messages.push(ChatMessage::aurora(
            response_text,
            intent,
            response_token_words,
            signals,
            eval_score,
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
                    ui.add_space(12.0);

                    // Tab Chat
                    let chat_active = self.view == AppView::Chat;
                    if ui.add(
                        egui::Button::new(
                            RichText::new("Chat")
                                .color(if chat_active { PURPLE_ACCENT } else { TEXT_DIM })
                                .font(FontId::new(12.0, FontFamily::Monospace))
                        )
                        .fill(if chat_active { BG_WIDGET } else { Color32::TRANSPARENT })
                        .rounding(Rounding::same(6.0))
                    ).clicked() { self.view = AppView::Chat; }

                    // Tab Training
                    let train_active = self.view == AppView::Training;
                    if ui.add(
                        egui::Button::new(
                            RichText::new("Training")
                                .color(if train_active { PURPLE_ACCENT } else { TEXT_DIM })
                                .font(FontId::new(12.0, FontFamily::Monospace))
                        )
                        .fill(if train_active { BG_WIDGET } else { Color32::TRANSPARENT })
                        .rounding(Rounding::same(6.0))
                    ).clicked() { self.view = AppView::Training; }

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
                            RichText::new(self.status.chars().take(80).collect::<String>())
                                .color(GREEN_OK)
                                .font(FontId::new(10.0, FontFamily::Monospace))
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

        // Wyślij wiadomość (tylko w widoku Chat)
        if send && !self.input.trim().is_empty() && self.view == AppView::Chat {
            let text = self.input.clone();
            self.input.clear();
            self.process_input(&text);
        }

        // Główny panel
        egui::CentralPanel::default()
            .frame(Frame::none().fill(BG_DARKEST).inner_margin(Margin::symmetric(16.0, 12.0)))
            .show(ctx, |ui| {
                match self.view {
                    AppView::Chat => {
                        ScrollArea::vertical()
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                let msgs = self.messages.clone();
                                for msg in &msgs {
                                    self.render_message(ui, msg);
                                    ui.add_space(8.0);
                                }
                            });
                    }
                    AppView::Training => {
                        self.render_training_panel(ui);
                    }
                }
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

                    // Eval score
                    if let Some(score) = msg.eval_score {
                        let score_color = if score > 0.6 { GREEN_OK }
                            else if score > 0.35 { ORANGE_WARN }
                            else { Color32::from_rgb(255, 80, 80) };
                        ui.label(
                            RichText::new(format!("eval: {:.2}", score))
                                .color(score_color)
                                .font(FontId::new(10.0, FontFamily::Monospace))
                        );
                    }
                }
            }
        });
    }

    fn render_training_panel(&mut self, ui: &mut egui::Ui) {
        use eframe::egui::Frame;

        ui.label(
            RichText::new("Training Panel")
                .color(TEXT_BRIGHT)
                .font(FontId::new(18.0, FontFamily::Monospace))
                .strong()
        );
        ui.add_space(8.0);

        // Statystyki
        let stats = self.behavior.training.stats_summary();
        ui.label(RichText::new(&stats).color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
        ui.add_space(12.0);

        ScrollArea::vertical().show(ui, |ui| {
            // --- DODAJ PRZYKŁAD ---
            Frame::none().fill(BG_PANEL)
                .rounding(Rounding::same(8.0))
                .stroke(Stroke::new(1.0, PURPLE_DIM))
                .inner_margin(Margin::same(12.0))
                .show(ui, |ui| {
                    ui.label(RichText::new("ADD TRAINING EXAMPLE").color(PURPLE_LIGHT)
                        .font(FontId::new(13.0, FontFamily::Monospace)).strong());
                    ui.separator();
                    ui.add_space(4.0);

                    egui::Grid::new("train_grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                        ui.label(RichText::new("Input:").color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::singleline(&mut self.train_input)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0)
                            .hint_text("what is your name"));
                        ui.end_row();

                        ui.label(RichText::new("Intent:").color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::singleline(&mut self.train_intent)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(PURPLE_LIGHT).desired_width(200.0)
                            .hint_text("QUERY_ENTITY"));
                        ui.end_row();

                        ui.label(RichText::new("Expected output:").color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::multiline(&mut self.train_output)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0).desired_rows(3)
                            .hint_text("I am Aurora..."));
                        ui.end_row();

                        ui.label(RichText::new("Key concepts:").color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::singleline(&mut self.train_concepts)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0)
                            .hint_text("name, identity, aurora (comma separated)"));
                        ui.end_row();
                    });

                    ui.add_space(8.0);
                    if ui.add(
                        egui::Button::new(RichText::new("+ Add Example").color(TEXT_BRIGHT)
                            .font(FontId::new(12.0, FontFamily::Monospace)))
                            .fill(PURPLE_MID).rounding(Rounding::same(6.0))
                            .min_size(Vec2::new(140.0, 30.0))
                    ).clicked() && !self.train_input.is_empty() && !self.train_output.is_empty() {
                        let concepts: Vec<&str> = self.train_concepts.split(',')
                            .map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                        self.behavior.training.add_example(
                            &self.train_input.clone(),
                            &self.train_intent.clone(),
                            &self.train_output.clone(),
                            concepts,
                        );
                        self.train_input.clear();
                        self.train_output.clear();
                        self.train_concepts.clear();
                    }
                });

            ui.add_space(12.0);

            // --- KOREKCJA ---
            Frame::none().fill(BG_PANEL)
                .rounding(Rounding::same(8.0))
                .stroke(Stroke::new(1.0, PURPLE_DIM))
                .inner_margin(Margin::same(12.0))
                .show(ui, |ui| {
                    ui.label(RichText::new("CORRECT A RESPONSE").color(ORANGE_WARN)
                        .font(FontId::new(13.0, FontFamily::Monospace)).strong());
                    ui.separator();
                    ui.add_space(4.0);

                    egui::Grid::new("correction_grid").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                        ui.label(RichText::new("Input:").color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::singleline(&mut self.correction_input)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0));
                        ui.end_row();

                        ui.label(RichText::new("Bad response:").color(Color32::from_rgb(255,80,80))
                            .font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::multiline(&mut self.correction_bad)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0).desired_rows(2));
                        ui.end_row();

                        ui.label(RichText::new("Correct response:").color(GREEN_OK)
                            .font(FontId::new(11.0, FontFamily::Monospace)));
                        ui.add(TextEdit::multiline(&mut self.correction_good)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY).desired_width(400.0).desired_rows(2));
                        ui.end_row();
                    });

                    ui.add_space(8.0);
                    if ui.add(
                        egui::Button::new(RichText::new("✓ Save Correction").color(TEXT_BRIGHT)
                            .font(FontId::new(12.0, FontFamily::Monospace)))
                            .fill(Color32::from_rgb(80, 40, 0)).rounding(Rounding::same(6.0))
                            .min_size(Vec2::new(160.0, 30.0))
                    ).clicked() && !self.correction_input.is_empty() && !self.correction_good.is_empty() {
                        let inp  = self.correction_input.clone();
                        let bad  = self.correction_bad.clone();
                        let good = self.correction_good.clone();
                        self.behavior.training.add_correction(&inp, &bad, &good, "UNKNOWN");
                        self.behavior.training.save_corrections("aurora_corrections.log");
                        self.correction_input.clear();
                        self.correction_bad.clear();
                        self.correction_good.clear();
                    }
                });

            ui.add_space(12.0);

            // --- LISTA PRZYKŁADÓW ---
            Frame::none().fill(BG_PANEL)
                .rounding(Rounding::same(8.0))
                .stroke(Stroke::new(1.0, PURPLE_DIM))
                .inner_margin(Margin::same(12.0))
                .show(ui, |ui| {
                    ui.label(RichText::new(
                        format!("EXAMPLES ({} total)", self.behavior.training.examples.len()))
                        .color(PURPLE_LIGHT).font(FontId::new(13.0, FontFamily::Monospace)).strong());
                    ui.separator();
                    ui.add_space(4.0);

                    let examples = self.behavior.training.examples.clone();
                    for ex in &examples {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("▸").color(PURPLE_MID));
                            ui.label(RichText::new(&ex.input).color(TEXT_PRIMARY)
                                .font(FontId::new(11.0, FontFamily::Monospace)));
                            ui.label(RichText::new(format!("→ {}", ex.expected_intent))
                                .color(TEXT_DIM).font(FontId::new(10.0, FontFamily::Monospace)));
                        });
                    }
                });
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