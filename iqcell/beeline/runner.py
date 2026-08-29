"""Wrap invocations of a cloned Murali-group/Beeline repository.

BEELINE runs GRN inference algorithms through Docker containers and evaluates
their ranked-edge outputs against a ground-truth network. This module shells
out to a local Beeline checkout's ``BLRunner.py`` / ``BLEvaluator.py`` entry
points, with an availability probe that never itself invokes Docker.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List


class BeelineNotAvailableError(RuntimeError):
    """Raised when the Beeline repo (or Docker) is not usable for a run."""


_SETUP_GUIDANCE = (
    "BEELINE is not available. Clone https://github.com/Murali-group/Beeline, "
    "set up its conda env (bash utils/setupAnacondaVENV.sh), pull algorithm "
    "images (bash utils/initialize.sh), ensure Docker is running, and pass the "
    "repo path to BeelineRunner(beeline_repo=...)."
)


@dataclass
class BeelineStatus:
    """Result of :meth:`BeelineRunner.check_available` (never raises)."""

    available: bool
    has_blrunner: bool
    has_docker: bool
    message: str


class BeelineRunner:
    """Drive a local Beeline checkout via subprocess.

    Parameters
    ----------
    beeline_repo:
        Path to a cloned Beeline repository (contains ``BLRunner.py``).
    python_exe:
        Python interpreter used to launch BEELINE's entry points. Defaults to
        ``"python"`` (typically the activated ``BEELINE`` conda env).
    """

    def __init__(self, beeline_repo: str, python_exe: str = "python") -> None:
        self.beeline_repo = beeline_repo
        self.python_exe = python_exe

    def check_available(self) -> BeelineStatus:
        """Probe whether BEELINE can run. Does not raise and does not run Docker.

        Only performs filesystem existence checks and ``shutil.which`` lookups.
        """
        has_blrunner = os.path.isfile(os.path.join(self.beeline_repo, "BLRunner.py"))
        has_docker = shutil.which("docker") is not None
        available = has_blrunner and has_docker

        if available:
            message = "BEELINE repo and Docker detected."
        elif not os.path.isdir(self.beeline_repo):
            message = f"Beeline repo path does not exist: {self.beeline_repo}"
        elif not has_blrunner:
            message = f"BLRunner.py not found under {self.beeline_repo}"
        else:
            message = "Docker executable not found on PATH."

        return BeelineStatus(
            available=available,
            has_blrunner=has_blrunner,
            has_docker=has_docker,
            message=message,
        )

    def _require_available(self) -> None:
        status = self.check_available()
        if not status.available:
            raise BeelineNotAvailableError(f"{status.message}\n{_SETUP_GUIDANCE}")

    def _invoke(self, argv: List[str]) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            argv,
            cwd=self.beeline_repo,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"BEELINE command failed (exit {proc.returncode}): "
                f"{' '.join(argv)}\nstderr:\n{proc.stderr}"
            )
        return proc

    def run(self, config_path: str) -> subprocess.CompletedProcess:
        """Run inference algorithms: ``python BLRunner.py -c <config>``."""
        self._require_available()
        argv = [self.python_exe, "BLRunner.py", "-c", os.path.abspath(config_path)]
        return self._invoke(argv)

    def evaluate(
        self, config_path: str, auc: bool = True, epr: bool = True
    ) -> subprocess.CompletedProcess:
        """Evaluate results: ``python BLEvaluator.py -c <config> [-a] [-e]``."""
        self._require_available()
        argv = [self.python_exe, "BLEvaluator.py", "-c", os.path.abspath(config_path)]
        if auc:
            argv.append("-a")
        if epr:
            argv.append("-e")
        return self._invoke(argv)
