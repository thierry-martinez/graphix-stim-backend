"""Uncorrelated depolarising noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from graphix.channels import KrausChannel, KrausData
from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import (
    ApplyNoise,
    Noise,
    NoiseModel,
)
from graphix.ops import Ops
from typing_extensions import assert_never, override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import CommandOrNoise
    from numpy.random import Generator


@dataclass(frozen=True)
class SinglePauliNoise(Noise):
    """One-qubit depolarising noise with probabibity `prob`."""

    prob: float
    error_type: Literal["X", "Z"] = "X"

    @property
    @override
    def nqubits(self) -> int:
        return 1

    @override
    def to_kraus_channel(self) -> KrausChannel:
        if self.error_type == "Z":
            return KrausChannel([KrausData(self.prob, Ops.Z)])

        return KrausChannel([KrausData(self.prob, Ops.X)])


@dataclass(frozen=True)
class SinglePauliNoiseModel(NoiseModel):
    """
    Test noise model on 3 qubit line graph and deterministic X or Z on middle qubit.

    edges: list of possible edges to draw from
    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    prob: float
    error_type: Literal["X", "Z"] = "X"

    @override
    def input_nodes(
        self, nodes: Iterable[int], rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        """Return the noise to apply to input nodes."""
        return []

    @override
    def command(
        self, cmd: CommandOrNoise, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> list[CommandOrNoise]:
        # flag to check of target node '1' has been visited to not apply noise twice
        match cmd.kind:
            case (
                CommandKind.N
                | CommandKind.M
                | CommandKind.X
                | CommandKind.Z
                | CommandKind.C
                | CommandKind.T
                | CommandKind.ApplyNoise
                | CommandKind.S
            ):
                return [cmd]
            case CommandKind.E:
                if 0 in cmd.nodes and 1 in cmd.nodes:
                    return [
                        cmd,
                        ApplyNoise(
                            noise=SinglePauliNoise(
                                self.prob, self.error_type
                            ),  # another thing where str is not subtype of Literal?
                            nodes=[1],
                        ),
                    ]
                return [cmd]
            case _:
                assert_never(cmd.kind)

    @override
    def confuse_result(
        self, cmd: BaseM, result: Outcome, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> Outcome:
        return result
