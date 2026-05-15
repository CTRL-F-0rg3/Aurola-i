// src/aurora_gui.rs
// GUI kalibracji AI "Aurora" — egui, ciemny fioletowy motyw
//
// Cargo.toml dependencies:
// eframe = "0.27"
// egui = "0.27"
// serde = { version = "1", features = ["derive"] }
// serde_json = "1"

use eframe::egui;
use eframe::egui::{
    Color32, FontId, Frame, Margin, Rounding, Stroke, Vec2,
    FontFamily, RichText, ScrollArea, Slider, TextEdit,
};
use serde::{Deserialize, Serialize};

// ========================
// PALETA KOLORÓW
// ========================

const BG_DARKEST:   Color32 = Color32::from_rgb(10,  6,  20);
const BG_DARK:      Color32 = Color32::from_rgb(18, 12, 35);
const BG_PANEL:     Color32 = Color32::from_rgb(25, 16, 50);
const BG_WIDGET:    Color32 = Color32::from_rgb(35, 22, 68);
const BG_HOVER:     Color32 = Color32::from_rgb(50, 32, 95);

const PURPLE_LIGHT: Color32 = Color32::from_rgb(180, 130, 255);
const PURPLE_MID:   Color32 = Color32::from_rgb(140,  80, 230);
const PURPLE_ACCENT:Color32 = Color32::from_rgb(200, 100, 255);
const PURPLE_DIM:   Color32 = Color32::from_rgb( 90,  50, 150);

const TEXT_PRIMARY: Color32 = Color32::from_rgb(220, 200, 255);
const TEXT_DIM:     Color32 = Color32::from_rgb(130, 100, 180);
const TEXT_BRIGHT:  Color32 = Color32::from_rgb(255, 240, 255);

const GREEN_OK:     Color32 = Color32::from_rgb( 80, 220, 140);
const ORANGE_WARN:  Color32 = Color32::from_rgb(255, 160,  60);
const RED_ERR:      Color32 = Color32::from_rgb(255,  80,  80);

// ========================
// DANE STANU
// ========================

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuronLayer {
    pub name:    String,
    pub neurons: usize,
    pub active:  bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AuroraIdentity {
    pub name:        String,
    pub description: String,
    pub language:    String,
    pub version:     String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CalibrationConfig {
    pub layers:          Vec<NeuronLayer>,
    pub max_token_input: usize,
    pub max_token_output: usize,
    pub temperature:     f32,
    pub identity:        AuroraIdentity,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                NeuronLayer { name: "Input Layer".into(),   neurons: 512,  active: true },
                NeuronLayer { name: "Hidden Layer 1".into(),neurons: 1024, active: true },
                NeuronLayer { name: "Hidden Layer 2".into(),neurons: 1024, active: true },
                NeuronLayer { name: "Hidden Layer 3".into(),neurons: 512,  active: true },
                NeuronLayer { name: "Output Layer".into(),  neurons: 256,  active: true },
            ],
            max_token_input:  512,
            max_token_output: 256,
            temperature:      0.7,
            identity: AuroraIdentity {
                name:        "Aurora".into(),
                description: "".into(),
                language:    "pl+en".into(),
                version:     "0.1.0".into(),
            },
        }
    }
}

// ========================
// STAN APLIKACJI
// ========================

pub struct AuroraApp {
    config:        CalibrationConfig,
    active_tab:    Tab,
    test_input:    String,
    test_output:   String,
    status_msg:    String,
    status_ok:     bool,
    add_layer_name: String,
}

#[derive(PartialEq)]
enum Tab {
    Identity,
    Neurons,
    Grammar,
    Test,
}

impl Default for AuroraApp {
    fn default() -> Self {
        Self {
            config:         CalibrationConfig::default(),
            active_tab:     Tab::Identity,
            test_input:     String::new(),
            test_output:    String::new(),
            status_msg:     "Gotowy".into(),
            status_ok:      true,
            add_layer_name: String::new(),
        }
    }
}

// ========================
// STYL egui
// ========================

fn apply_aurora_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.visuals.window_fill        = BG_DARK;
    style.visuals.panel_fill         = BG_DARKEST;
    style.visuals.faint_bg_color     = BG_PANEL;
    style.visuals.extreme_bg_color   = BG_DARKEST;

    style.visuals.widgets.inactive.bg_fill   = BG_WIDGET;
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, PURPLE_DIM);
    style.visuals.widgets.inactive.rounding  = Rounding::same(6.0);

    style.visuals.widgets.hovered.bg_fill    = BG_HOVER;
    style.visuals.widgets.hovered.fg_stroke  = Stroke::new(1.5, PURPLE_LIGHT);
    style.visuals.widgets.hovered.rounding   = Rounding::same(6.0);

    style.visuals.widgets.active.bg_fill     = PURPLE_MID;
    style.visuals.widgets.active.fg_stroke   = Stroke::new(2.0, PURPLE_ACCENT);
    style.visuals.widgets.active.rounding    = Rounding::same(6.0);

    style.visuals.widgets.open.bg_fill       = BG_HOVER;

    style.visuals.selection.bg_fill          = PURPLE_MID;
    style.visuals.selection.stroke           = Stroke::new(1.0, PURPLE_LIGHT);

    style.visuals.window_stroke  = Stroke::new(1.0, PURPLE_DIM);
    style.visuals.window_rounding = Rounding::same(10.0);

    style.spacing.item_spacing   = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(12.0, 6.0);

    ctx.set_style(style);
}

// ========================
// HELPERY UI
// ========================

fn section_frame() -> Frame {
    Frame {
        fill:         BG_PANEL,
        rounding:     Rounding::same(8.0),
        stroke:       Stroke::new(1.0, PURPLE_DIM),
        inner_margin: Margin::same(12.0),
        outer_margin: Margin::symmetric(0.0, 4.0),
        ..Default::default()
    }
}

fn header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text)
            .color(PURPLE_LIGHT)
            .font(FontId::new(14.0, FontFamily::Monospace))
            .strong()
    );
    ui.separator();
    ui.add_space(2.0);
}

fn label_dim(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(TEXT_DIM).font(FontId::new(11.0, FontFamily::Monospace)));
}

fn neuron_bar(ui: &mut egui::Ui, count: usize, max: usize) {
    let fill = (count as f32 / max as f32).clamp(0.0, 1.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 6.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, Rounding::same(3.0), BG_WIDGET);
    let mut fill_rect = rect;
    fill_rect.max.x = rect.min.x + rect.width() * fill;
    painter.rect_filled(fill_rect, Rounding::same(3.0), PURPLE_MID);
}

// ========================
// IMPLEMENTACJA APP
// ========================

impl eframe::App for AuroraApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        apply_aurora_style(ctx);

        // Topbar
        egui::TopBottomPanel::top("topbar")
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
                        RichText::new("  AI Calibration System")
                            .color(TEXT_DIM)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let status_color = if self.status_ok { GREEN_OK } else { RED_ERR };
                        ui.label(
                            RichText::new(&self.status_msg)
                                .color(status_color)
                                .font(FontId::new(11.0, FontFamily::Monospace))
                        );
                        ui.label(RichText::new("◉ ").color(status_color));
                    });
                });
            });

        // Sidebar z tabami
        egui::SidePanel::left("sidebar")
            .frame(Frame::none().fill(BG_DARK).inner_margin(Margin::same(8.0)))
            .exact_width(160.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                self.tab_button(ui, Tab::Identity, "⬡  Tożsamość");
                self.tab_button(ui, Tab::Neurons,  "⬡  Neurony");
                self.tab_button(ui, Tab::Grammar,  "⬡  Gramatyka");
                self.tab_button(ui, Tab::Test,     "⬡  Test");

                ui.add_space(16.0);
                ui.separator();
                ui.add_space(8.0);

                // Podsumowanie neuronów
                label_dim(ui, "Łącznie neuronów:");
                let total: usize = self.config.layers.iter()
                    .filter(|l| l.active)
                    .map(|l| l.neurons)
                    .sum();
                ui.label(
                    RichText::new(format!("{}", total))
                        .color(PURPLE_LIGHT)
                        .font(FontId::new(18.0, FontFamily::Monospace))
                        .strong()
                );

                ui.add_space(8.0);
                label_dim(ui, "Warstwy aktywne:");
                let active = self.config.layers.iter().filter(|l| l.active).count();
                ui.label(
                    RichText::new(format!("{}/{}", active, self.config.layers.len()))
                        .color(PURPLE_LIGHT)
                        .font(FontId::new(14.0, FontFamily::Monospace))
                );

                ui.add_space(16.0);

                // Przycisk zapisu
                if ui.add(
                    egui::Button::new(
                        RichText::new("💾 Zapisz config")
                            .color(TEXT_BRIGHT)
                            .font(FontId::new(11.0, FontFamily::Monospace))
                    )
                    .fill(PURPLE_MID)
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(144.0, 32.0))
                ).clicked() {
                    self.save_config();
                }

                ui.add_space(4.0);

                if ui.add(
                    egui::Button::new(
                        RichText::new("📂 Wczytaj config")
                            .color(TEXT_DIM)
                            .font(FontId::new(11.0, FontFamily::Monospace))
                    )
                    .fill(BG_WIDGET)
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(144.0, 28.0))
                ).clicked() {
                    self.load_config();
                }
            });

        // Główny panel
        egui::CentralPanel::default()
            .frame(Frame::none().fill(BG_DARKEST).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    match self.active_tab {
                        Tab::Identity => self.panel_identity(ui),
                        Tab::Neurons  => self.panel_neurons(ui),
                        Tab::Grammar  => self.panel_grammar(ui),
                        Tab::Test     => self.panel_test(ui),
                    }
                });
            });
    }
}

impl AuroraApp {
    fn tab_button(&mut self, ui: &mut egui::Ui, tab: Tab, label: &str) {
        let active = self.active_tab == tab;
        let color  = if active { PURPLE_ACCENT } else { TEXT_DIM };
        let fill   = if active { BG_WIDGET } else { Color32::TRANSPARENT };

        if ui.add(
            egui::Button::new(
                RichText::new(label)
                    .color(color)
                    .font(FontId::new(12.0, FontFamily::Monospace))
            )
            .fill(fill)
            .rounding(Rounding::same(6.0))
            .min_size(Vec2::new(144.0, 28.0))
        ).clicked() {
            self.active_tab = tab;
        }
    }

    // ========================
    // PANEL: TOŻSAMOŚĆ
    // ========================
    fn panel_identity(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Tożsamość AI")
                .color(TEXT_BRIGHT)
                .font(FontId::new(18.0, FontFamily::Monospace))
                .strong()
        );
        ui.add_space(12.0);

        section_frame().show(ui, |ui| {
            header(ui, "PODSTAWOWE INFORMACJE");

            egui::Grid::new("identity_grid")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    label_dim(ui, "Nazwa:");
                    ui.add(
                        TextEdit::singleline(&mut self.config.identity.name)
                            .font(FontId::new(13.0, FontFamily::Monospace))
                            .text_color(PURPLE_LIGHT)
                            .desired_width(280.0)
                    );
                    ui.end_row();

                    label_dim(ui, "Wersja:");
                    ui.add(
                        TextEdit::singleline(&mut self.config.identity.version)
                            .font(FontId::new(13.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY)
                            .desired_width(280.0)
                    );
                    ui.end_row();

                    label_dim(ui, "Języki:");
                    ui.add(
                        TextEdit::singleline(&mut self.config.identity.language)
                            .font(FontId::new(13.0, FontFamily::Monospace))
                            .text_color(TEXT_PRIMARY)
                            .desired_width(280.0)
                    );
                    ui.end_row();
                });
        });

        ui.add_space(8.0);

        section_frame().show(ui, |ui| {
            header(ui, "OPIS / OSOBOWOŚĆ");
            label_dim(ui, "Co Aurora wie o sobie:");
            ui.add_space(4.0);
            ui.add(
                TextEdit::multiline(&mut self.config.identity.description)
                    .font(FontId::new(12.0, FontFamily::Monospace))
                    .text_color(TEXT_PRIMARY)
                    .desired_width(f32::INFINITY)
                    .desired_rows(8)
                    .hint_text("Wpisz opis tożsamości, wartości, zachowania Aurory...")
            );
        });
    }

    // ========================
    // PANEL: NEURONY
    // ========================
    fn panel_neurons(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Kalibracja Neuronów")
                .color(TEXT_BRIGHT)
                .font(FontId::new(18.0, FontFamily::Monospace))
                .strong()
        );
        ui.add_space(12.0);

        // Ustawienia tokenów
        section_frame().show(ui, |ui| {
            header(ui, "TOKENY");
            egui::Grid::new("token_grid")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    label_dim(ui, "Max tokenów wejście:");
                    ui.add(Slider::new(&mut self.config.max_token_input, 64..=4096)
                        .text("")
                        .clamp_to_range(true)
                    );
                    ui.end_row();

                    label_dim(ui, "Max tokenów wyjście:");
                    ui.add(Slider::new(&mut self.config.max_token_output, 64..=4096)
                        .text("")
                        .clamp_to_range(true)
                    );
                    ui.end_row();

                    label_dim(ui, "Temperatura:");
                    ui.add(Slider::new(&mut self.config.temperature, 0.0..=2.0)
                        .text("")
                        .clamp_to_range(true)
                    );
                    ui.end_row();
                });
        });

        ui.add_space(8.0);

        // Warstwy
        section_frame().show(ui, |ui| {
            header(ui, "WARSTWY NEURONÓW");

            let max_neurons = 2048usize;
            let mut to_remove: Option<usize> = None;

            for (i, layer) in self.config.layers.iter_mut().enumerate() {
                ui.add_space(4.0);

                // Nagłówek warstwy
                ui.horizontal(|ui| {
                    let dot_color = if layer.active { GREEN_OK } else { TEXT_DIM };
                    ui.label(RichText::new("●").color(dot_color));
                    ui.add(
                        TextEdit::singleline(&mut layer.name)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                            .text_color(PURPLE_LIGHT)
                            .desired_width(160.0)
                    );
                    ui.checkbox(&mut layer.active, "");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(
                            egui::Button::new(RichText::new("✕").color(RED_ERR).font(FontId::new(10.0, FontFamily::Monospace)))
                                .fill(Color32::TRANSPARENT)
                                .rounding(Rounding::same(4.0))
                        ).clicked() {
                            to_remove = Some(i);
                        }
                        ui.label(
                            RichText::new(format!("{} n", layer.neurons))
                                .color(TEXT_DIM)
                                .font(FontId::new(11.0, FontFamily::Monospace))
                        );
                    });
                });

                // Suwak neuronów
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.add(
                        Slider::new(&mut layer.neurons, 1..=max_neurons)
                            .text("")
                            .clamp_to_range(true)
                    );
                });

                // Pasek wizualny
                ui.add_space(2.0);
                neuron_bar(ui, layer.neurons, max_neurons);
                ui.add_space(4.0);
            }

            if let Some(idx) = to_remove {
                self.config.layers.remove(idx);
            }

            // Dodaj warstwę
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add(
                    TextEdit::singleline(&mut self.add_layer_name)
                        .font(FontId::new(12.0, FontFamily::Monospace))
                        .hint_text("Nazwa nowej warstwy...")
                        .desired_width(200.0)
                );
                if ui.add(
                    egui::Button::new(
                        RichText::new("+ Dodaj warstwę")
                            .color(TEXT_BRIGHT)
                            .font(FontId::new(11.0, FontFamily::Monospace))
                    )
                    .fill(PURPLE_MID)
                    .rounding(Rounding::same(6.0))
                ).clicked() && !self.add_layer_name.is_empty() {
                    self.config.layers.push(NeuronLayer {
                        name:    self.add_layer_name.clone(),
                        neurons: 256,
                        active:  true,
                    });
                    self.add_layer_name.clear();
                }
            });
        });
    }

    // ========================
    // PANEL: GRAMATYKA
    // ========================
    fn panel_grammar(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Gramatyka")
                .color(TEXT_BRIGHT)
                .font(FontId::new(18.0, FontFamily::Monospace))
                .strong()
        );
        ui.add_space(12.0);

        section_frame().show(ui, |ui| {
            header(ui, "GENEROWANE PRZEZ PYTHON");
            label_dim(ui, "Pliki generowane automatycznie:");
            ui.add_space(6.0);

            let files = [
                ("grammar_svo.asm",    "Reguły SVO (Subject-Verb-Object)", GREEN_OK),
                ("grammar_np.asm",     "Frazy nominalne (NP -> DET ADJ N)", GREEN_OK),
                ("grammar_vp.asm",     "Frazy werbalne (VP -> V NP)",       ORANGE_WARN),
                ("grammar_endings.asm","Końcówki odmian",                    ORANGE_WARN),
            ];

            for (file, desc, color) in &files {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("◈").color(*color));
                    ui.label(
                        RichText::new(*file)
                            .color(PURPLE_LIGHT)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                    );
                    ui.label(
                        RichText::new(format!("— {}", desc))
                            .color(TEXT_DIM)
                            .font(FontId::new(11.0, FontFamily::Monospace))
                    );
                });
            }

            ui.add_space(12.0);

            if ui.add(
                egui::Button::new(
                    RichText::new("▶ Uruchom generator gramatyki")
                        .color(TEXT_BRIGHT)
                        .font(FontId::new(12.0, FontFamily::Monospace))
                )
                .fill(PURPLE_MID)
                .rounding(Rounding::same(6.0))
                .min_size(Vec2::new(260.0, 34.0))
            ).clicked() {
                self.run_grammar_generator();
            }
        });

        ui.add_space(8.0);

        section_frame().show(ui, |ui| {
            header(ui, "REGUŁY SVO");
            label_dim(ui, "Podstawowa kolejność: Subject → Verb → Object");
            ui.add_space(6.0);

            let rules = [
                ("S → NP VP",          "Zdanie = Fraza nom. + Fraza verb."),
                ("NP → DET ADJ N",     "Fraza nom. = Det. + Przym. + Rzecz."),
                ("NP → DET N",         "Fraza nom. = Det. + Rzecz."),
                ("VP → V NP",          "Fraza verb. = Czas. + Fraza nom."),
                ("VP → V ADV",         "Fraza verb. = Czas. + Przysł."),
            ];

            for (rule, desc) in &rules {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(*rule)
                            .color(PURPLE_LIGHT)
                            .font(FontId::new(12.0, FontFamily::Monospace))
                    );
                    ui.label(
                        RichText::new(format!("   {}", desc))
                            .color(TEXT_DIM)
                            .font(FontId::new(11.0, FontFamily::Monospace))
                    );
                });
            }
        });
    }

    // ========================
    // PANEL: TEST
    // ========================
    fn panel_test(&mut self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new("Test Tokenizacji")
                .color(TEXT_BRIGHT)
                .font(FontId::new(18.0, FontFamily::Monospace))
                .strong()
        );
        ui.add_space(12.0);

        section_frame().show(ui, |ui| {
            header(ui, "ZAPYTANIE");
            ui.add(
                TextEdit::multiline(&mut self.test_input)
                    .font(FontId::new(13.0, FontFamily::Monospace))
                    .text_color(TEXT_PRIMARY)
                    .desired_width(f32::INFINITY)
                    .desired_rows(4)
                    .hint_text("Wpisz tekst do tokenizacji...")
            );
            ui.add_space(8.0);

            if ui.add(
                egui::Button::new(
                    RichText::new("▶ Tokenizuj")
                        .color(TEXT_BRIGHT)
                        .font(FontId::new(12.0, FontFamily::Monospace))
                )
                .fill(PURPLE_MID)
                .rounding(Rounding::same(6.0))
                .min_size(Vec2::new(140.0, 32.0))
            ).clicked() {
                self.run_test_tokenize();
            }
        });

        ui.add_space(8.0);

        section_frame().show(ui, |ui| {
            header(ui, "WYNIK TOKENIZACJI");
            if self.test_output.is_empty() {
                label_dim(ui, "Brak wyników — wpisz tekst i kliknij Tokenizuj");
            } else {
                ui.add(
                    TextEdit::multiline(&mut self.test_output.clone())
                        .font(FontId::new(11.0, FontFamily::Monospace))
                        .text_color(GREEN_OK)
                        .desired_width(f32::INFINITY)
                        .desired_rows(12)
                        .interactive(false)
                );
            }
        });
    }

    // ========================
    // AKCJE
    // ========================

    fn save_config(&mut self) {
        match serde_json::to_string_pretty(&self.config) {
            Ok(json) => {
                match std::fs::write("aurora_config.json", &json) {
                    Ok(_) => {
                        self.status_msg = "Config zapisany -> aurora_config.json".into();
                        self.status_ok  = true;
                    }
                    Err(e) => {
                        self.status_msg = format!("Błąd zapisu: {}", e);
                        self.status_ok  = false;
                    }
                }
            }
            Err(e) => {
                self.status_msg = format!("Błąd serializacji: {}", e);
                self.status_ok  = false;
            }
        }
    }

    fn load_config(&mut self) {
        match std::fs::read_to_string("aurora_config.json") {
            Ok(json) => {
                match serde_json::from_str::<CalibrationConfig>(&json) {
                    Ok(cfg) => {
                        self.config     = cfg;
                        self.status_msg = "Config wczytany".into();
                        self.status_ok  = true;
                    }
                    Err(e) => {
                        self.status_msg = format!("Błąd parsowania: {}", e);
                        self.status_ok  = false;
                    }
                }
            }
            Err(e) => {
                self.status_msg = format!("Błąd odczytu: {}", e);
                self.status_ok  = false;
            }
        }
    }

    fn run_grammar_generator(&mut self) {
        match std::process::Command::new("python3")
            .arg("src/grammar_gen.py")
            .status()
        {
            Ok(s) if s.success() => {
                self.status_msg = "Generator gramatyki zakończony OK".into();
                self.status_ok  = true;
            }
            Ok(s) => {
                self.status_msg = format!("Generator zakończył się kodem: {:?}", s.code());
                self.status_ok  = false;
            }
            Err(e) => {
                self.status_msg = format!("Błąd uruchomienia: {}", e);
                self.status_ok  = false;
            }
        }
    }

    fn run_test_tokenize(&mut self) {
        // Prosta tokenizacja po stronie GUI (bez pełnego Rust tokenizera)
        // Docelowo można wywołać tokenizer.rs przez FFI lub IPC
        let words: Vec<&str> = self.test_input.split_whitespace().collect();
        let mut out = String::new();

        out.push_str(&format!(
            "{:<20} {:<8} {:<6} {:<10}\n",
            "SŁOWO", "TOKEN_ID", "POS", "STATUS"
        ));
        out.push_str(&"-".repeat(50));
        out.push('\n');

        for word in &words {
            let upper = word.to_uppercase();
            // Placeholder — docelowo lookup w vocab
            out.push_str(&format!(
                "{:<20} {:<8} {:<6} {}\n",
                word,
                "?",
                "?",
                if upper.len() > 1 { "lookup pending" } else { "SYMBOL" }
            ));
        }

        out.push_str(&format!("\nŁącznie słów: {}", words.len()));
        self.test_output = out;
        self.status_msg  = format!("Przetworzono {} słów", words.len());
        self.status_ok   = true;
    }
}

// ========================
// MAIN
// ========================

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Aurora — AI Calibration")
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([800.0, 500.0])
            .with_icon(eframe::icon_data::from_png_bytes(&[]).unwrap_or_default()),
        ..Default::default()
    };

    eframe::run_native(
        "Aurora",
        options,
        Box::new(|_cc| Box::new(AuroraApp::default())),
    )
}
