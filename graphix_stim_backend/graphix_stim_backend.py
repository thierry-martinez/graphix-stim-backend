"""Stim backend for Graphix."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import stim
from graphix import Pattern, command
from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.fundamentals import ANGLE_PI, Axis, Plane, Sign
from graphix.measurements import Measurement, Outcome, PauliMeasurement, outcome
from graphix.noise_models.depolarising import DepolarisingNoise, TwoQubitDepolarisingNoise
from graphix.optimization import StandardizedPattern
from graphix.sim.base_backend import Backend, Matrix
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.states import BasicStates, PlanarState, State
from typing_extensions import assert_never, override

from graphix_stim_backend.single_pauli_noise_model import SinglePauliNoise

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from typing import TypeAlias

    from graphix.command import CommandType, Node
    from graphix.noise_models.noise_model import ApplyNoise, CommandOrNoise, NoiseModel
    from graphix.sim.data import Data
    from numpy.random import Generator

    GraphType: TypeAlias = nx.Graph[int]
else:
    GraphType = nx.Graph


class BasicState(Enum):
    """Enumeration for basic states."""

    ZERO = PlanarState(Plane.XZ, 0)
    ONE = PlanarState(Plane.XZ, ANGLE_PI)
    PLUS = PlanarState(Plane.XY, 0)
    MINUS = PlanarState(Plane.XY, ANGLE_PI)
    PLUS_I = PlanarState(Plane.XY, ANGLE_PI / 2)
    MINUS_I = PlanarState(Plane.XY, -ANGLE_PI / 2)

    @staticmethod
    def try_from_statevector(sv: Matrix) -> BasicState | None:
        """Return the BasicState corresponding to the parameter, or not if it is not a basic state."""
        return next((bs for bs in BasicState if np.all(bs.value.to_statevector() == sv)), None)

    @staticmethod
    def try_from_state(s: State) -> BasicState | None:
        """Return the BasicState corresponding to the parameter, or not if it is not a basic state."""
        if isinstance(s, PlanarState):
            return next((bs for bs in BasicState if bs.value == s), None)
        return BasicState.try_from_statevector(s.to_statevector())


BASIC_STATE_TO_CLIFFORD = {
    BasicState.ZERO: [Clifford.Z],
    BasicState.ONE: [Clifford.X],
    BasicState.PLUS: [Clifford.H],
    BasicState.MINUS: [Clifford.H, Clifford.Z],
    BasicState.PLUS_I: [Clifford.H, Clifford.S],
    BasicState.MINUS_I: [Clifford.H, Clifford.S, Clifford.Z],
}

X: Literal["X"] = "X"
Z: Literal["Z"] = "Z"


def get_stabilizers(graph: GraphType) -> list[stim.PauliString]:
    """
    Generate the canonical stabilizers for a given graph state.

    :param graph: graph state
    :return: list of stim.Paulistring containing len(nodes) stabilizers
    """

    def get_stabilizer_for_node(node: int) -> stim.PauliString:
        ps = stim.PauliString(graph.number_of_nodes())
        ps[node] = X
        for k in graph.neighbors(node):
            ps[k] = Z
        return ps

    return [get_stabilizer_for_node(node) for node in graph.nodes]


def apply_clifford(sim: stim.TableauSimulator, node: int, clifford: Clifford) -> None:
    """Apply Clifford gate to a tableau simulator."""
    match clifford:
        case Clifford.H:
            sim.h(node)
        case Clifford.S:
            sim.s(node)
        case Clifford.SDG:
            sim.s_dag(node)
        case Clifford.Z:
            sim.z(node)
        case clifford.X:
            sim.x(node)
        case _:
            for h_s_z in clifford.hsz:
                match h_s_z:
                    case Clifford.H:
                        sim.h(node)
                    case Clifford.S:
                        sim.s(node)
                    case Clifford.Z:
                        sim.z(node)
                    case _:
                        msg = "Unreachable"
                        raise ValueError(msg)


def pauli_measurement_to_clifford_gates(
    measurement: PauliMeasurement,
) -> list[Clifford]:
    """Enumerate Clifford gates for a Pauli measurement."""
    match measurement.sign:
        case Sign.PLUS:
            match measurement.axis:
                case Axis.X:
                    return [Clifford.H]
                case Axis.Y:
                    return [Clifford.H, Clifford.S]
                case Axis.Z:
                    return []
        case Sign.MINUS:
            match measurement.axis:
                case Axis.X:
                    return [Clifford.H, Clifford.Z]
                case Axis.Y:
                    return [Clifford.H, Clifford.S, Clifford.Z]
                case Axis.Z:
                    return [Clifford.X]


def apply_pauli_measurement(  # noqa: PLR0913
    sim: stim.TableauSimulator,
    node: int,
    measurement: PauliMeasurement,
    branch: Mapping[int, Outcome] | None = None,
    *,
    s_signal: bool,
    t_signal: bool,
) -> Outcome:
    """Apply a Pauli measurement to a tableau simulator."""
    if s_signal:
        sim.h(node)
        sim.z(node)
        sim.h(node)
    if t_signal:
        sim.z(node)
    cliffords = pauli_measurement_to_clifford_gates(measurement)
    for clifford in reversed(cliffords):
        apply_clifford(sim, node, clifford.conj)
    branch_result = None if branch is None else branch.get(node)
    if branch_result is None:
        result = outcome(sim.measure(node))
    else:
        result = branch_result
        sim.postselect_z(node, desired_value=(result == 1))
    for clifford in cliffords:
        apply_clifford(sim, node, clifford)
    return result


@dataclass
class RenumberedGraph:
    """Renumbering of a graph."""

    nodes: list[int]
    edges: list[tuple[int, int]]
    renumbering: dict[int, int]
    graph: GraphType


def get_renumbered_graph(pattern: Pattern) -> RenumberedGraph:
    """
    Compute the graph state where nodes are indexed with a range of integers starting from 0.

    :param pattern: pattern
    :return: the renumbering and the graph
    """
    graph = pattern.extract_graph()
    nodes = list(graph.nodes())
    renumbering = {node: i for i, node in enumerate(graph.nodes())}
    renumbered_edges = [(renumbering[u], renumbering[v]) for (u, v) in graph.edges()]
    renumbered_graph = GraphType()
    renumbered_graph.add_nodes_from(range(len(nodes)))
    renumbered_graph.add_edges_from(renumbered_edges)
    return RenumberedGraph(nodes, list(graph.edges()), renumbering, renumbered_graph)


def _graph_state_to_edges_and_vops(
    graph_state: stim.Circuit,
) -> tuple[list[tuple[int, int]], dict[int, Clifford]]:
    edges: list[tuple[int, int]] = []
    vops: dict[int, Clifford] = {}
    # "Circuit" has no attribute "__iter__"
    # (but __len__ and __getitem__)
    instruction: stim.CircuitInstruction
    for instruction in graph_state:  # type: ignore[attr-defined]
        match instruction.name:
            case "RX":
                pass
            case "CZ":
                edges.extend((u.value, v.value) for u, v in instruction.target_groups())
            case "H" | "S" | "X" | "Y" | "Z":
                clifford: Clifford = getattr(Clifford, instruction.name)
                for (u,) in instruction.target_groups():
                    vops[u.value] = clifford @ vops.get(u.value, Clifford.I)
            case "TICK":
                pass
            case _:
                raise ValueError(instruction.name)
    return edges, vops


def cut_pattern(pattern: Pattern) -> tuple[Pattern, Pattern]:
    """Cut pattern in a Clifford part and a non-Clifford part."""
    pauli_pattern = Pattern(input_nodes=pattern.input_nodes)
    first_non_pauli = None
    it = iter(pattern)
    for cmd in it:
        if cmd.kind == CommandKind.M and not isinstance(cmd.measurement, PauliMeasurement):
            first_non_pauli = cmd
            break
        pauli_pattern.add(cmd)
    non_pauli_pattern = Pattern(input_nodes=pauli_pattern.output_nodes)
    if first_non_pauli is not None:
        non_pauli_pattern.add(first_non_pauli)
    non_pauli_pattern.extend(it)
    return (pauli_pattern, non_pauli_pattern)


@dataclass(frozen=True)
class _AbstractStimBackend(Backend[stim.TableauSimulator]):
    state: stim.TableauSimulator = dataclasses.field(init=False, default_factory=stim.TableauSimulator)
    branch: dict[int, Outcome] | None = None

    @override
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        state = BasicState.try_from_statevector(Statevec(data).psi)

        if state is None:
            msg = f"Incorrect state value: stim can only prepare stabiliser states {data}."
            raise ValueError(msg)

        if state == BasicState.ZERO:
            # required by stim otherwise empty stabiliser
            self.state.z(*nodes)
        elif state == BasicState.ONE:
            self.state.x(*nodes)
        elif state == BasicState.PLUS:
            self.state.h(*nodes)
        elif state == BasicState.MINUS:
            self.state.h(*nodes)
            self.state.z(*nodes)
        elif state == BasicState.PLUS_I:
            self.state.h(*nodes)
            self.state.s(*nodes)
        elif state == BasicState.MINUS_I:
            self.state.h(*nodes)
            self.state.s(*nodes)
            self.state.z(*nodes)
        else:
            assert_never(state)

    @override
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        self.state.cz(*edge)

    @override
    def measure(
        self, node: int, measurement: Measurement, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        if not isinstance(measurement, PauliMeasurement):
            msg = f"The measurement {measurement} is not Pauli."
            raise TypeError(msg)
        return apply_pauli_measurement(self.state, node, measurement, self.branch, s_signal=False, t_signal=False)

    @override
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        apply_clifford(self.state, node, clifford)

    @override
    def apply_noise(self, cmd: ApplyNoise) -> None:
        match cmd.noise:
            case DepolarisingNoise(prob=prob):
                (q,) = cmd.nodes
                self.state.depolarize1(q, p=prob)
            case TwoQubitDepolarisingNoise(prob=prob):
                (q0, q1) = cmd.nodes
                self.state.depolarize2(q0, q1, p=prob)
            # add case here
            case _:
                msg = f"Unsupported noise: {cmd.noise} and {cmd.nodes}"
                raise ValueError(msg)

    @override
    def correct_byproduct(self, cmd: command.X | command.Z) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""
        clifford = Clifford.X if cmd.kind == CommandKind.X else Clifford.Z
        self.apply_clifford(node=cmd.node, clifford=clifford)

    @override
    def finalize(self, output_nodes: Iterable[int]) -> None:
        pass

    def to_pattern(self, input_nodes: list[int], output_nodes: list[int]) -> Pattern:
        """Return a pattern implementing the simulation."""
        tableau = self.state.current_inverse_tableau().inverse()
        circuit = tableau.to_circuit("graph_state")
        return _graph_state_to_pattern(circuit, input_nodes, output_nodes)


class StimBackend(_AbstractStimBackend):
    """Stim backend."""

    def __init__(self, branch: Mapping[int, Outcome] | None = None) -> None:
        """Initialize Stim backend."""
        super().__init__(branch=None if branch is None else dict(branch))


def pattern_to_stim_circuit(  # noqa: C901,PLR0912,PLR0915
    pattern: Pattern,
    noise_model: NoiseModel | None = None,
    input_state: dict[int, BasicState] | BasicState = BasicState.PLUS,
    fixed_states: dict[int, BasicState] | None = None,
) -> tuple[stim.Circuit, dict[int, int]]:
    """Convert a pattern to a Stim circuit."""
    circuit = stim.Circuit()
    if isinstance(input_state, BasicState):
        for clifford in BASIC_STATE_TO_CLIFFORD[input_state]:
            circuit.append(str(clifford), targets=pattern.input_nodes)  # type: ignore[call-overload]
    else:
        other_nodes = set(input_state.keys()) - set(pattern.input_nodes)
        if other_nodes:
            msg = f"Not input states: {other_nodes}"
            raise ValueError(msg)
        for node in pattern.input_nodes:
            basic_state = input_state[node]
            for clifford in BASIC_STATE_TO_CLIFFORD[basic_state]:
                circuit.append(str(clifford), targets=[node])  # type: ignore[call-overload]
    if noise_model is None:
        actual_pattern: list[CommandOrNoise] = list(pattern)
    else:
        actual_pattern = noise_model.input_nodes(pattern.input_nodes)
        actual_pattern.extend(noise_model.transpile(list(pattern)))
    measure_count = 0
    measure_indices: dict[int, int] = {}

    def get_target(node: int) -> stim.GateTarget:
        return stim.target_rec(measure_indices[node] - measure_count)

    for cmd in actual_pattern:
        if cmd.kind == CommandKind.N:
            basic_state_or_none = None if fixed_states is None else fixed_states.get(cmd.node)
            if basic_state_or_none is None:
                basic_state_or_none = BasicState.try_from_statevector(Statevec(cmd.state).psi)
                if basic_state_or_none is None:
                    msg = f"Non-Pauli preparation: {cmd}"
                    raise ValueError(msg)
            for clifford in BASIC_STATE_TO_CLIFFORD[basic_state_or_none]:
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.E:
            circuit.append("CZ", cmd.nodes)  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.M:
            for node in cmd.s_domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
            for node in cmd.t_domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
            if not isinstance(cmd.measurement, PauliMeasurement):
                msg = f"Non-Pauli measurement: {cmd}"
                raise ValueError(msg)
            cliffords = pauli_measurement_to_clifford_gates(cmd.measurement)
            for clifford in reversed(cliffords):
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            circuit.append("M", [cmd.node])  # type: ignore[call-overload]
            for clifford in cliffords:
                circuit.append(str(clifford), [cmd.node])  # type: ignore[call-overload]
            measure_indices[cmd.node] = measure_count
            measure_count += 1
        elif cmd.kind == CommandKind.X:
            for node in cmd.domain:
                circuit.append("CX", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.Z:
            for node in cmd.domain:
                circuit.append("CZ", [get_target(node), cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.C:
            circuit.append(str(cmd.clifford), [cmd.node])  # type: ignore[call-overload]
        elif cmd.kind == CommandKind.ApplyNoise:
            match cmd.noise:
                case DepolarisingNoise(prob=prob):
                    (q,) = cmd.nodes
                    circuit.append("DEPOLARIZE1", [q], prob)
                case TwoQubitDepolarisingNoise(prob=prob):
                    (q0, q1) = cmd.nodes
                    circuit.append("DEPOLARIZE2", [q0, q1], prob)
                # add case here
                case SinglePauliNoise(prob=prob, error_type=et):  #  case SinglePauliNoise(error_type='X')
                    (q,) = cmd.nodes
                    if et == "X":
                        circuit.append("X_ERROR", [q], prob)
                    elif et == "Z":
                        circuit.append("Z_ERROR", [q], prob)
                    else:
                        msg = f"Unsupported single-Pauli: {et} and {cmd.nodes}"
                        raise ValueError(msg)
                case _:
                    msg = f"Unsupported noise: {cmd.noise} and {cmd.nodes}"
                    raise ValueError(msg)

    return circuit, measure_indices


def _graph_state_to_pattern(circuit: stim.Circuit, input_nodes: list[int], output_nodes: list[int]) -> Pattern:
    edges, vops = _graph_state_to_edges_and_vops(circuit)
    pattern = Pattern(input_nodes)
    input_node_set = set(input_nodes)
    output_node_set = set(output_nodes)
    pattern.extend(command.N(node=node) for node in output_nodes if node not in input_node_set)
    pattern.extend(command.E(nodes=nodes) for nodes in edges)
    pattern.extend(
        command.C(node=node, clifford=clifford) for node, clifford in vops.items() if node in output_node_set
    )
    return pattern


def incorporate_pauli_results(  # noqa: C901, PLR0912
    target: Pattern, commands: Iterable[CommandType], results: Mapping[Node, Outcome]
) -> None:
    """Return an equivalent pattern where results from Pauli presimulation are integrated in corrections."""
    for cmd in commands:
        match cmd.kind:
            case CommandKind.M:
                s = _incorporate_pauli_results_in_domain(results, cmd.s_domain)
                t = _incorporate_pauli_results_in_domain(results, cmd.t_domain)
                if s or t:
                    if s:
                        apply_x, new_s_domain = s
                    else:
                        apply_x = False
                        new_s_domain = cmd.s_domain
                    if t:
                        apply_z, new_t_domain = t
                    else:
                        apply_z = False
                        new_t_domain = cmd.t_domain
                    new_cmd = command.M(cmd.node, cmd.measurement, new_s_domain, new_t_domain)
                    if apply_x:
                        new_cmd = new_cmd.clifford(Clifford.X)
                    if apply_z:
                        new_cmd = new_cmd.clifford(Clifford.Z)
                    target.add(new_cmd)
                else:
                    target.add(cmd)
            case CommandKind.X | CommandKind.Z:
                signal = _incorporate_pauli_results_in_domain(results, cmd.domain)
                if signal:
                    apply_c, new_domain = signal
                    if new_domain:
                        cmd_cstr = command.X if cmd.kind == CommandKind.X else command.Z
                        target.add(cmd_cstr(cmd.node, new_domain))
                    if apply_c:
                        c = Clifford.X if cmd.kind == CommandKind.X else Clifford.Z
                        target.add(command.C(cmd.node, c))
                else:
                    target.add(cmd)
            case _:
                target.add(cmd)


def _incorporate_pauli_results_in_domain(
    results: Mapping[Node, Outcome], domain: AbstractSet[int]
) -> tuple[bool, set[int]] | None:
    if not (results.keys() & domain):
        return None
    new_domain = set(domain - results.keys())
    odd_outcome = sum(outcome for node, outcome in results.items() if node in domain) % 2
    return odd_outcome == 1, new_domain


@dataclass(frozen=True, slots=True)
class PresimulatedPattern:
    """Pattern with presimulation results."""

    pattern: Pattern
    results: Mapping[Node, Outcome]


def presimulate_pauli(
    pattern: Pattern, *, branch: Mapping[int, Outcome] | None = None, leave_input: bool = False
) -> PresimulatedPattern:
    """Return a pattern where Clifford measurements have been presimulated."""
    leave_nodes = set(pattern.input_nodes) if leave_input else None
    pattern = StandardizedPattern.from_pattern(pattern).perform_pauli_pushing(leave_nodes).to_pattern()
    pauli_pattern, non_pauli_pattern = cut_pattern(pattern)
    backend = StimBackend(branch=branch)
    measure_method = DefaultMeasureMethod()
    pauli_pattern.simulate_pattern(backend, measure_method=measure_method)
    output_node_set = set(pauli_pattern.output_nodes)
    input_nodes = [node for node in pattern.input_nodes if node in output_node_set]
    result_pattern = backend.to_pattern(input_nodes, non_pauli_pattern.input_nodes)
    incorporate_pauli_results(result_pattern, non_pauli_pattern, measure_method.results)
    result_pattern.reorder_output_nodes(pattern.output_nodes)
    return PresimulatedPattern(result_pattern, measure_method.results)
