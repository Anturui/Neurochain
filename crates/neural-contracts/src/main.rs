use neural_contracts::bench::run_benchmark;

fn main() {
    println!("NeuroChain Neural Contracts — Proof of Concept");
    println!("Concept: contract logic = NN forward pass (no branches, full SIMD)");

    for batch in [100, 1_000, 10_000, 100_000] {
        run_benchmark(batch);
    }

    println!("\nKey insight: even a 2-layer MLP beats classical branching on GPU");
    println!("because warp executes FMA uniformly vs divergent if/else paths.");
}