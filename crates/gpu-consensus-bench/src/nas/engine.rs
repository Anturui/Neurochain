//! Continuous NAS engine stub

use super::ArchitectureGene;
use std::sync::Arc;
use tokio::sync::RwLock;
use ndarray_rand::rand::{Rng, thread_rng};


pub struct ContinuousNASEngine {
    population: Vec<ArchitectureGene>,
    best_architecture: Option<ArchitectureGene>,
    running: Arc<RwLock<bool>>,
}

impl ContinuousNASEngine {
    pub fn new(population_size: usize) -> Self {

        let mut rng = rand::rng();
        
        let population = (0..population_size)
            .map(|_| ArchitectureGene::random(&mut rng))
            .collect();

        Self {
            population,
            best_architecture: None,
            running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&mut self) {  // &mut self вместо &self
        *self.running.write().await = true;
    }

    pub async fn stop(&self) {
        *self.running.write().await = false;
    }
    
    pub fn get_best_proposal(&self) -> Option<ArchitectureGene> {
        self.best_architecture.clone()
    }
}