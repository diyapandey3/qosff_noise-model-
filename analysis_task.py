"""
This project involves creating a noise generator and seeing its effect on quantum circuits using various algorithms. 
Noise model represents the erros that can occur duing simaultions and quantum computations
Pauli Errors: Represents diffrenet types of quantum error. 
Sevela functions here does specific tasks: 
1. Function add_noise_to_circuit creates a noise model and applies correspinding Pauli errors to it. 
2. Function transform_to_gate_basis converts quantum circuits to specific gate basis given CX, ID, RZ, SX, X and more. 
3. Quantum Fourier Transform QFT, helps in computing, processing and transforming qunatum states using Hadamard gates H, Controlled Phase Rotations cp, swaps. 
4. Quantum Addition Draper Adder, quantum_sum function implements the Draper adder algorithm using QFT. 
5. Function analyze_noise_effects analyses how noise impacts the outcomes of quantum addition. 
6. Function plot_results helps to visualize that using matplotlib
"""
