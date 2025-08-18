"""Run tests with nox."""

from __future__ import annotations

import nox
from nox import Session


def install_pytest(session: Session) -> None:
    """Install pytest when requirements-dev.txt is not installed."""
    session.install("pytest", "pytest-mock", "psutil")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install("-e", ".")
    install_pytest(session)
    session.run("pytest")
