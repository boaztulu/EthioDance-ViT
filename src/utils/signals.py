"""SIGUSR1 handler for graceful SLURM requeue.

SLURM delivers SIGUSR1 ahead of the time limit when the job is started with
`--signal=USR1@<seconds>` (see scripts/train_hipergator.sh). We catch it, flip
a flag that the trainer polls each step, write a resumable checkpoint, and then
call `scontrol requeue $SLURM_JOB_ID` so the job picks up where it left off on
the next allocation.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess

log = logging.getLogger(__name__)

# Windows lacks SIGUSR1; guard at module load so imports work cross-platform.
_SIGUSR1 = getattr(signal, "SIGUSR1", None)
_SIGTERM = getattr(signal, "SIGTERM", None)
_DEFAULT_SIGNALS = tuple(s for s in (_SIGUSR1, _SIGTERM) if s is not None)


class RequeueHandler:
    def __init__(self, signals_to_handle: tuple[int, ...] | None = None) -> None:
        self.should_stop = False
        self.should_requeue = False
        self._installed_signals = signals_to_handle or _DEFAULT_SIGNALS

    def install(self) -> None:
        for sig in self._installed_signals:
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # Not supported on this platform / subthread.
                pass

    def _handle(self, sig: int, _frame) -> None:
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        log.warning("Received %s — flagging graceful stop/requeue.", name)
        self.should_stop = True
        if _SIGUSR1 is not None and sig == _SIGUSR1:
            self.should_requeue = True

    def maybe_requeue(self) -> None:
        if not self.should_requeue:
            return
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            log.warning("No SLURM_JOB_ID in env; skipping scontrol requeue.")
            return
        log.warning("Requeuing SLURM job %s.", job_id)
        try:
            subprocess.run(["scontrol", "requeue", job_id], check=False)
        except FileNotFoundError:
            log.warning("scontrol not found; requeue requires running under SLURM.")
