"""
Statevector Simulator - Core Selection Problem Statement 1

Build a statevector simulator for an n-qubit system using NumPy.

Your simulator must support:
    - Single qubit gates: X, H, Z
    - Two qubit gates: CNOT, CZ
    - Half entropy of the system: the von-Neumann entanglement entropy

You may ONLY use NumPy (and the Python standard library). No Qiskit,
no other quantum computing libraries.
"""

import numpy as np

X = np.array([[0,1],[1,0]], dtype=complex)
H = (1/np.sqrt(2))* np.array([[1,1],[1,-1]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
CZ = np.diag([1,1,1,-1]).astype(complex)
                
class StatevectorSimulator:
    """A statevector simulator for an n-qubit quantum system."""

    def __init__(self, num_qubits: int):
        """
        Initialize the simulator in the |0...0> state.

        Args:
            num_qubits: Number of qubits n. The statevector must be a
                complex NumPy array of size 2**n.
        """
        self.num_qubits = num_qubits
        self.state = None  # initialize to |0...0>
        self.reset()

    def apply_gate(self, U: np.ndarray, qubits:list) -> None:
        # Applies 2^k x 2^k unitary U to given list of k qubits
        # Works for any k by reshaping to n-index tensor, moving target qubit axes to the front, contracting with U then moving the axes back
        n = self.num_qubits
        k = len(qubits)
        psi = self.state.reshape([2]*n)
        psi = np.moveaxis(psi,qubits,range(k))
        psi = psi.reshape(2**k,-1)
        psi = U @ psi
        psi = psi.reshape([2]*k + [2]*(n-k))
        psi = np.moveaxis(psi,range(k),qubits)
        self.state = psi.reshape(-1)



    def x(self, qubit: int) -> None:
        """Apply the Pauli-X (NOT) gate to the given qubit."""
        self.apply_gate(X,[qubit])
        # raise NotImplementedError

    def h(self, qubit: int) -> None:
        """Apply the Hadamard gate to the given qubit."""
        self.apply_gate(H,[qubit])
        # raise NotImplementedError

    def z(self, qubit: int) -> None:
        """Apply the Pauli-Z gate to the given qubit."""
        self.apply_gate(Z,[qubit])
        # raise NotImplementedError

    def cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate with the given control and target qubits."""
        self.apply_gate(CNOT,[control,target])
        # raise NotImplementedError

    def cz(self, control: int, target: int) -> None:
        """Apply a controlled-Z gate with the given control and target qubits."""
        self.apply_gate(CZ,[control,target])
        # raise NotImplementedError

    def half_entropy(self) -> float:
        """
        Compute the entanglement entropy across the half bipartition.

        Returns:
            The half-cut entanglement entropy in bits.
        """
        n = self.num_qubits
        s = n // 2
        a = self.state.reshape(2**s,2**(n-s))
        _, lambdas, _ = np.linalg.svd(a)
        prob = lambdas**2
        prob = prob[prob>0]
        return float(-np.sum(prob * np.log2(prob)))
        # raise NotImplementedError

    def get_statevector(self) -> np.ndarray:
        """Return the current statevector as a NumPy array."""
        return self.state
        # raise NotImplementedError

    def get_probabilities(self) -> np.ndarray:
        """Return the current probabilities as a Numpy array"""
        return np.abs(self.state)**2
        # raise NotImplementedError

    def reset(self) -> None:
        """Reset the simulator back to the |0...0> state."""
        self.state = np.zeros(2**self.num_qubits,dtype=complex)
        self.state[0] = 1.0
        # raise NotImplementedError

    def oracle_2qubit(self, marked_state: int) -> None:
        #phase flip amplitude of |marked_state> using X and CZ
        #wrap marked basis state in X gate so that it is mapped to |11>, apply CZ then undo X gates
        bits = [(marked_state >> (self.num_qubits - 1 - i)) & 1 for i in range(self.num_qubits)]
        flipped = [i for i,b in enumerate(bits) if b == 0]
        for i in flipped:
            self.x(i)

        self.cz(0,1)
        for i in flipped:
            self.x(i)
        

    def grover_2qubit(self, marked_state: int) -> None:
        """
        Run Grover's search algorithm on 2 qubits using your simulator.

        The marked state should have probability ~1 after one iteration.

        Args:
            marked_state: Index (0 to 3) of the state the oracle marks.
        """
        if self.num_qubits != 2:
            raise ValueError("grover_2qubit requires a 2-qubit simulator")
        self.reset()
        self.h(0)
        self.h(1)
        self.oracle_2qubit(marked_state)
        self.h(0)
        self.h(1)
        self.x(0)
        self.x(1)
        self.cz(0,1)
        self.x(0)
        self.x(1)
        self.h(0)
        self.h(1)

        # raise NotImplementedError


if __name__ == "__main__":
    # Run 2-qubit Grover search marking the state |11> (index 3).
    simulator = StatevectorSimulator(2)
    marked_state = 3
    simulator.grover_2qubit(marked_state)
    final_state = simulator.get_statevector()
    probabilities = simulator.get_probabilities()

    print("Final statevector:", final_state)
    print("Probabilities:", probabilities)
    print("Measured state:", np.argmax(probabilities))
