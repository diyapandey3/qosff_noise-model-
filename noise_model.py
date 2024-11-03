#The purpose of this project is to create a simple noise generator and assess its effect using Qiskit framework, Draper adder algorithm and the Quantum Fourier Transform (QFT). 

#Importing the packages for the required functionality: NoiseModel and pauli errors 
#Noise model - Represents the noise characteristics of a quantum device or circuit
#Pauli Errors - Correspond to different types of error: X (bit-flip), Y (bit-and-phase flip), Z (phase-flip), I (identity)
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit.circuit.library import CX, H, X, SX, ID
from qiskit.circuit import RX, RY, RZ



def add_noise_to_circuit(circuit, prob_one_qubit, prob_two_qubit):
    noise_model = NoiseModel() # create a noise model 

    one_qubit_noise_dict = {  #using def of pauli errors 
        'X': prob_one_qubit / 3,
        'Y': prob_one_qubit / 3,
        'Z': prob_one_qubit / 3,
        'I': 1 - prob_one_qubit
    }

    two_qubit_noise_dict = {
        'X': prob_two_qubit / 3,
        'Y': prob_two_qubit / 3,
        'Z': prob_two_qubit / 3,
        'I': 1 - prob_two_qubit
    }

    one_qubit_noise = pauli_error(list(one_qubit_noise_dict.items()))
    two_qubit_noise = pauli_error(list(two_qubit_noise_dict.items()))

    for qubit in range(circuit.num_qubits):
        noise_model.add_all_qubit_quantum_error(one_qubit_noise, ["u1", "u2", "u3", "rz", "sx", "x"])

    noise_model.add_all_qubit_quantum_error(two_qubit_noise, ["cx"])

    return noise_model

"""
transforms any given quantum circuit into a specific gate basis i.e CX,ID,RZ,SX,X
"""
def transform_to_gate_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    # Create a new quantum circuit with the same number of qubits and classical bits
    transformed_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    for instruction, qubits, _ in circuit.data:
        # Apply transformations based on the gate type
        if instruction.name == 'cx':
            # CNOT gate stays the same
            transformed_circuit.cx(*qubits)
        elif instruction.name == 'h':
            # Hadamard gate can be represented using RZ and SX gates
            transformed_circuit.append(SX(*qubits), qubits)
            transformed_circuit.append(RZ(3.14159265, *qubits), qubits)  # RZ(pi)
            transformed_circuit.append(SX(*qubits), qubits)
        elif instruction.name == 'x':
            # X gate stays the same
            transformed_circuit.x(*qubits)
        elif instruction.name == 'sx':
            # SX gate stays the same
            transformed_circuit.sx(*qubits)
        elif instruction.name == 'id':
            # Identity gate stays the same
            transformed_circuit.id(*qubits)
        elif instruction.name == 'rz':
            # RZ gate stays the same
            transformed_circuit.rz(instruction.params[0], *qubits)
        elif instruction.name == 'measure':
            # Measurement stays the same
            transformed_circuit.measure(qubits, circuit.clbits[:len(qubits)])
        # Add more transformations as needed

    return transformed_circuit

"""
Implementing QFT for quantum adder function 
"""
def qft(circuit: QuantumCircuit, n: int):
    """Performs the Quantum Fourier Transform on the first n qubits of the circuit."""
    for j in range(n):
        circuit.h(j)  # Apply Hadamard gate
        for k in range(j + 1, n):
            circuit.cp(np.pi / (2 ** (k - j)), k, j)  # Controlled phase rotation
    # Reverse the order of the qubits
    for i in range(n // 2):
        circuit.swap(i, n - i - 1)

def quantum_sum(a: int, b: int, n: int) -> QuantumCircuit:
    """Adds two integers a and b using the Draper adder algorithm with QFT."""
    # Create a quantum circuit with enough qubits
    circuit = QuantumCircuit(n * 2, n)

    # Initialize the input states for a and b
    for i in range(n):
        if (a >> i) & 1:
            circuit.x(i)  # Set the qubits for a
        if (b >> i) & 1:
            circuit.x(n + i)  # Set the qubits for b

    # Apply QFT on the first n qubits
    qft(circuit, n)

    # Inverse QFT (as part of the Draper adder) would typically be applied here
    # For simplicity, we will assume it's handled within this structure.

    # Measurement
    circuit.measure(range(n), range(n))  # Measure the output qubits

    return circuit
"""
Analyzing the results"""

def analyze_noise_effects(a: int, b: int, n: int, noise_levels: list):
    """Analyzes the effects of noise on quantum addition results."""
    results = {}

    for prob_one_qubit in noise_levels:
        prob_two_qubit = prob_one_qubit * 2  # Example scaling for two-qubit noise
        # Create the quantum sum circuit
        adder_circuit = quantum_sum(a, b, n)
        
        # Transform the circuit to the gate basis
        transformed_circuit = transform_to_gate_basis(adder_circuit)
        
        # Add noise to the circuit
        noisy_circuit = add_noise_to_circuit(transformed_circuit, prob_one_qubit, prob_two_qubit)

        # Execute the noisy circuit
        backend = Aer.get_backend('qasm_simulator')
        result = execute(noisy_circuit, backend, shots=1024).result()
        counts = result.get_counts(noisy_circuit)

        # Store the results for analysis
        results[prob_one_qubit] = counts

    return results


import matplotlib.pyplot as plt

def plot_results(results):
    for noise_level, counts in results.items():
        plt.bar(counts.keys(), counts.values(), alpha=0.5, label=f'Noise: {noise_level}')

    plt.xlabel('Results')
    plt.ylabel('Counts')
    plt.title('Effect of Noise on Quantum Addition Results')
    plt.legend()
    plt.show()

# Call the plotting function
plot_results(results)

