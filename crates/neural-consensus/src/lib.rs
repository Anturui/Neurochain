//! BFT Consensus for 10 neural validators
//! 
//! Key innovation: validators agree on NN outputs, not just state roots.
//! This requires deterministic inference (FP16 with fixed rounding modes).

pub mod bft;
pub mod validator_node;
pub mod state;

pub use bft::NeuroBft;
pub use validator_node::ValidatorNode;