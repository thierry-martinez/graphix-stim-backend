# Stim backend for Graphix

`graphix-stim-backend` is a plugin for the
[Graphix](https://github.com/TeamGraphix/graphix) library that
efficiently simulates Clifford patterns using the
[Stim](https://github.com/quantumlib/Stim) library.

It is distributed as a separate plugin because it depends on Stim.

## Installation

```bash
pip install https://github.com/thierry-martinez/graphix-stim-backend.git
```

## Usage

### Simulate a Clifford pattern

```python
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix_stim_backend import StimBackend
pattern = Pattern(input_nodes=[0])
pattern.add(command.N(node=1))
pattern.add(command.E(nodes=(0, 1)))
pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
pattern.simulate_pattern(backend=StimBackend())
```

### Simulate a Clifford pattern with noise

```python
from numpy.random import Generator
from graphix.noise_models.depolarising import DepolarisingNoiseModel
rng = Generator()
noise_model = DepolarisingNoiseModel(measure_error_prob=rng.random())
pattern.simulate_pattern(backend=StimBackend(), noise_model=noise_model)
```

### Presimulate the Clifford part of a pattern with Stim

```python
from numpy.random import Generator
from graphix.random_objects import rand_circuit
from graphix_stim_backend import presimulate_pauli
nqubits = 4
depth = 4
rng = Generator()
circuit = rand_circuit(nqubits, depth, rng)
pattern = circuit.transpile().pattern
pattern2 = presimulate_pauli(pattern, leave_input=False)
```

### Simulate a Clifford pattern with many shots

```python
from graphix import Pattern, command
from graphix.fundamentals import Plane
from graphix_stim_backend import BasicState, pattern_to_stim_circuit
pattern = Pattern(input_nodes=[0])
pattern.add(command.N(node=1))
pattern.add(command.E(nodes=(0, 1)))
pattern.add(command.M(node=0, plane=Plane.XY, angle=0.5))
stim_circuit, measure_indices = pattern_to_stim_circuit(
    pattern,
    input_state={0: BasicState.ZERO},
)
sample = stim_circuit.compile_sampler().sample(shots=1000)
```
