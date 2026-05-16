#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: u64,
    pub neuron_type: NeuronType,
    pub active: bool,

    pub activation: f32,
    pub previous_activation: f32,
    pub internal_state: f32,
    pub confidence: f32,

    pub inputs: Vec<f32>,
    pub weights: Vec<f32>,
    pub input_sources: Vec<u64>,

    pub output: f32,
    pub last_output: f32,
    pub output_strength: f32,

    pub connections: Vec<Connection>,

    pub activation_function: ActivationFunction,
    pub combination_method: CombinationMethod,
    pub threshold: f32,

    pub memory_trace: Vec<f32>,
    pub activation_history: Vec<f32>,
    pub decay: f32,

    pub learning_rate: f32,
    pub update_rule: UpdateRule,

    pub context_embedding: Vec<f32>,
    pub context_focus: f32,
    pub context_influence: f32,

    pub last_log: String,
    pub last_used_timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub target_id: u64,
    pub weight: f32,
    pub connection_type: ConnectionType,
}

#[derive(Debug, Clone)]
pub enum NeuronType {
    Feature,
    Routing,
    Memory,
    Decision,
}

#[derive(Debug, Clone)]
pub enum ConnectionType {
    Excitatory,
    Inhibitory,
    Modulatory,
}

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

#[derive(Debug, Clone)]
pub enum CombinationMethod {
    Sum,
    WeightedSum,
    Attention,
}

#[derive(Debug, Clone)]
pub enum UpdateRule {
    None,
    Hebbian,
    Custom,
}

