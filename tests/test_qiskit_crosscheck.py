"""
Qiskit/Aer cross-check: validate the analytic Ramsey formula against
a quantum circuit simulation with a phase-damping channel.

The test constructs the Ramsey circuit:
    |0⟩ → H → Rz(ωτ) → phase-damping(λ) → H → measure

where λ = 1 - C(τ)² = 1 - exp(-2(τ/T2*)^p).

P(0) from the circuit should match the analytic formula within 3σ
(binomial standard error) for ~200k shots.

Skipped automatically if qiskit or qiskit-aer is not installed.
"""
from __future__ import annotations
import pytest
import numpy as np

SHOTS = 200_000
TAU_VALUES = [0.5, 2.0, 8.0, 16.0]   # µs
T2_US = 10.0
OMEGA = 0.7                            # rad/µs
VISIBILITY = 0.9                       # not directly used in circuit; circuit uses ideal contrast
SHAPE = 1.0                            # exponential only (phase-damping matches p=1)


def _circuit_p0(tau_us: float, omega: float, t2_us: float, shots: int):
    """Build and run the Ramsey circuit; return outcome-0 frequency."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(1, 1)
    qc.h(0)                              # |0⟩ → |+⟩
    qc.rz(omega * tau_us, 0)             # phase accumulation
    # Phase damping: λ = 1 - exp(-2*(τ/T2*)^p)
    lam = float(1.0 - np.exp(-2.0 * (tau_us / t2_us) ** SHAPE))
    lam = np.clip(lam, 0.0, 1.0)
    qc.reset(0)  # placeholder; we apply the damping via Kraus

    # Rebuild without reset — use Kraus operators for phase damping
    qc2 = QuantumCircuit(1, 1)
    qc2.h(0)
    qc2.rz(omega * tau_us, 0)
    from qiskit.circuit.library import PhaseGate
    # Qiskit phase damping channel via save_density_matrix + Kraus isn't trivial;
    # use a noise model on the simulator instead.
    qc2.h(0)
    qc2.measure(0, 0)

    from qiskit_aer.noise import NoiseModel, phase_damping_error
    noise_model = NoiseModel()
    error = phase_damping_error(lam)
    # Apply phase damping before the final H (between the two H gates)
    # We insert it as a 1-qubit gate error on 'h' for the *second* h gate.
    # Cleaner: add phase damping as a general Kraus gate before final H.
    # Simplest valid approach: build a 3-instruction circuit with a noise gate.
    qc3 = QuantumCircuit(1, 1)
    qc3.h(0)
    qc3.rz(omega * tau_us, 0)
    # Apply noise on the identity step
    qc3.id(0)   # identity gate; we'll attach phase damping noise to 'id'
    qc3.h(0)
    qc3.measure(0, 0)

    nm = NoiseModel()
    nm.add_quantum_error(error, 'id', [0])

    sim = AerSimulator(noise_model=nm)
    job = sim.run(qc3, shots=shots)
    counts = job.result().get_counts()
    n0 = counts.get('0', 0)
    return n0 / shots


def test_qiskit_crosscheck():
    """P(0) from Qiskit Aer circuit matches analytic formula within 3σ."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    from core.physics import probability_zero

    for tau in TAU_VALUES:
        p_analytic = float(probability_zero(OMEGA, tau, T2_US, SHAPE, 1.0))
        # Note: circuit uses V=1 (ideal contrast); scale to compare
        p_circuit = _circuit_p0(tau, OMEGA, T2_US, SHOTS)

        sigma = np.sqrt(p_circuit * (1 - p_circuit) / SHOTS)
        z_score = abs(p_circuit - p_analytic) / (sigma + 1e-12)

        print(f"tau={tau:5.1f}us  analytic={p_analytic:.5f}  "
              f"circuit={p_circuit:.5f}  z={z_score:.2f}")

        assert z_score < 3.0, (
            f"τ={tau}µs: |P_circuit - P_analytic| = {abs(p_circuit - p_analytic):.5f} "
            f"({z_score:.1f}σ) — formula/circuit mismatch"
        )


def test_qiskit_available():
    """Just checks that qiskit and qiskit-aer can be imported."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit  # noqa: F401
    from qiskit_aer import AerSimulator  # noqa: F401
