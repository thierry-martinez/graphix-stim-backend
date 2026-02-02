"""Tests for Stim backend."""

import numpy as np
import pytest
import stim
from graphix import Circuit, Pattern, command
from graphix.branch_selector import (
    FixedBranchSelector,
    RandomBranchSelector,
)
from graphix.fundamentals import Plane
from graphix.noise_models.depolarising import DepolarisingNoiseModel
from graphix.noise_models.noise_model import NoiseModel
from graphix.optimization import StandardizedPattern
from graphix.random_objects import rand_circuit
from graphix.sim.base_backend import DenseState, Matrix, outer
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicStates
from numpy.random import PCG64, Generator

from graphix_stim_backend import (
    BasicState,
    StimBackend,
    cut_pattern,
    pattern_to_stim_circuit,
    presimulate_pauli,
)


def fidelity(u: Matrix, v: Matrix) -> float:
    """Compute fidelity between two vectors."""
    return np.abs(np.dot(u.conjugate(), v))  # type: ignore[no-any-return]


def compare_backend_results(state1: DenseState, state2: DenseState) -> float:
    """Compute fidelity between two backend states."""
    if isinstance(state1, Statevec) and isinstance(state2, Statevec):
        return fidelity(state1.flatten(), state2.flatten())
    if isinstance(state1, DensityMatrix):
        dm1 = state1
    elif isinstance(state1, Statevec):
        dm1 = DensityMatrix(state1)
    else:
        raise NotImplementedError
    if isinstance(state2, DensityMatrix):
        dm2 = state2
    elif isinstance(state2, Statevec):
        dm2 = DensityMatrix(state2)
    else:
        raise NotImplementedError
    return fidelity(dm1.rho.flatten(), dm2.rho.flatten())


def test_simple() -> None:
    """Test with simple pattern."""
    pattern = Pattern()
    pattern.add(command.N(node=0))
    pattern.add(command.N(node=1))
    pattern.add(command.N(node=2))
    pattern.add(command.E(nodes=(0, 1)))
    pattern.add(command.E(nodes=(1, 2)))
    pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
    pattern.add(command.M(node=1, plane=Plane.XY, angle=0.4, s_domain={0}))
    pattern2 = presimulate_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    pattern2.minimize_space()
    # Simulating the unprocessed pattern with the measures chosen by stim
    pbs = FixedBranchSelector(pattern2.results, RandomBranchSelector())
    # Instantiate the measure method to retrieve the measures of the non-Pauli nodes
    measure_method = DefaultMeasureMethod()
    state = pattern.simulate_pattern(branch_selector=pbs, measure_method=measure_method)
    # Simulating the processed pattern with the measures drawn for the previous simulation
    pbs2 = FixedBranchSelector(measure_method.results)
    state2 = pattern2.simulate_pattern(branch_selector=pbs2)
    assert compare_backend_results(state2, state) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_pauli_measurement_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    """Test with random circuits."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern2 = presimulate_pauli(pattern, leave_input=False)
    pattern.minimize_space()
    # pattern2.minimize_space()  # Break runnability!  # noqa: ERA001
    # Since the patterns are deterministic, we do not need to select a particular branch
    state = pattern.simulate_pattern()
    state2 = pattern2.simulate_pattern()
    assert compare_backend_results(state, state2) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_branch_selection(fx_bg: PCG64, jumps: int) -> None:
    """Test branch selection."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern_a = presimulate_pauli(pattern, leave_input=False)
    pattern_b = presimulate_pauli(pattern, leave_input=False, branch=pattern_a.results)
    assert list(pattern_a) == list(pattern_b)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_simulate_pauli_depolarising_noise(fx_bg: PCG64, jumps: int) -> None:
    """Test depolarising noise."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern = StandardizedPattern.from_pattern(pattern).perform_pauli_pushing().to_pattern()
    pauli_pattern, _non_pauli_pattern = cut_pattern(pattern)
    noise_model = DepolarisingNoiseModel()
    backend = StimBackend()
    pauli_pattern.simulate_pattern(backend, noise_model=noise_model)


def hpat() -> Pattern:
    """Return the Hadamard pattern."""
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


def simulate_with_noise_model_to_density_matrix(pattern: Pattern, noise_model: NoiseModel) -> Matrix:
    """Simulate noise with Stim and a density matrix."""
    backend = StimBackend()
    pattern.simulate_pattern(backend=backend, noise_model=noise_model)
    second_pattern = backend.to_pattern([], pattern.output_nodes)
    state = second_pattern.simulate_pattern()
    assert isinstance(state, Statevec)
    return outer(state.psi, state.psi.conj())


def test_noisy_measure_confuse_hadamard() -> None:
    """Test noise with Hadamard."""
    hadamard_pattern = hpat()
    noise_model = DepolarisingNoiseModel(measure_error_prob=1.0)
    rho = simulate_with_noise_model_to_density_matrix(hadamard_pattern, noise_model)
    # result should be |1>
    assert np.allclose(rho, np.array([[0.0, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("jumps", range(1, 11))
def test_noisy_measure_confuse_hadamard_random(fx_bg: PCG64, jumps: int) -> None:
    """Test random noise with Hadamard."""
    rng = Generator(fx_bg.jumped(jumps))
    hadamard_pattern = hpat()
    noise_model = DepolarisingNoiseModel(measure_error_prob=rng.random())
    rho = simulate_with_noise_model_to_density_matrix(hadamard_pattern, noise_model)
    assert np.allclose(rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
        rho,
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    )


def test_add_nodes() -> None:
    """Test adding nodes to Stim backend."""
    states = [
        BasicStates.ZERO,
        BasicStates.ONE,
        BasicStates.PLUS,
        BasicStates.MINUS,
        BasicStates.PLUS_I,
        BasicStates.MINUS_I,
    ]
    stabs = [stim.PauliString(s) for s in ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]]

    for i, state in enumerate(states):
        backend = StimBackend()
        backend.add_nodes([0], state)
        [
            stim_stab,
        ] = backend.state.canonical_stabilizers()
        assert stim_stab == stabs[i]


def test_pattern_to_stim_circuit(fx_rng: Generator) -> None:
    """Test pattern to Stim circuit conversion with random circuit."""
    nodes = 50
    planes = [Plane(p) for p in fx_rng.integers(low=1, high=4, size=nodes)]
    expected_results = [fx_rng.integers(2) == 1 for _ in range(nodes)]

    def get_input_state(node: int) -> BasicState:
        if planes[node] == Plane.XY:
            if expected_results[node]:
                return BasicState.MINUS
            return BasicState.PLUS
        if expected_results[node]:
            return BasicState.ONE
        return BasicState.ZERO

    pattern = Pattern(input_nodes=list(range(nodes)))
    node: int
    for node in fx_rng.choice(range(nodes), size=nodes, replace=False):
        pattern.add(command.M(node, plane=planes[node], angle=0))
    circuit, measure_indices = pattern_to_stim_circuit(
        pattern,
        input_state={node: get_input_state(node) for node in range(nodes)},
    )
    sample = circuit.compile_sampler().sample(shots=1000000)
    for shot in sample:
        assert [shot[measure_indices[i]] for i in range(nodes)] == expected_results


def test_pattern_to_stim_circuit_hadamard() -> None:
    """Test pattern to Stim circuit conversion with Hadamard."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.h(1)
    pattern = circuit.transpile().pattern
    node0 = pattern.output_nodes[0]
    node1 = pattern.output_nodes[1]
    pattern.add(command.M(node0, plane=Plane.XY))
    pattern.add(command.M(node1, plane=Plane.XY))
    stim_circuit, measure_indices = pattern_to_stim_circuit(
        pattern,
        input_state={0: BasicState.ZERO, 1: BasicState.ONE},
    )
    sample = stim_circuit.compile_sampler().sample(shots=1000)
    for s in sample:
        assert not s[measure_indices[node0]]
        assert s[measure_indices[node1]]
