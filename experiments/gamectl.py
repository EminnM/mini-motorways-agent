"""Start / stop dedicated game instances for the harness, reusing the existing launch logic in train_ppo.py
(identical command line, environment variables and readiness check). Never touches other instances."""
import os
import time

from . import paths
from . import config as cfgmod  # noqa: F401


class GameInstance:
    """Context manager: one headless (by default) game process with its own IPC dir and log."""

    def __init__(self, name: str, headless: bool = True):
        import train_ppo
        self._tp = train_ppo
        self.ipc_dir = os.path.join(paths.LOGS_DIR, f"ipc_{name}")
        self.log = os.path.join(paths.LOGS_DIR, f"instrumentation_{name}.log")
        self.inst = train_ppo.Inst(self.ipc_dir, self.log)
        self.inst.headless = headless

    def start(self, timeout: int = 300) -> "GameInstance":
        os.makedirs(self.ipc_dir, exist_ok=True)
        if not self._tp.launch_instances([self.inst], timeout):
            self.stop()
            raise RuntimeError(f"game instance {self.ipc_dir} did not become ready within {timeout}s")
        return self

    def stop(self):
        self._tp.kill_instance(self.inst)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def kill_by_log(ipc_dir: str, log: str):
    import train_ppo
    train_ppo.kill_instance(train_ppo.Inst(ipc_dir, log))
