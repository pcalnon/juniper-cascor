#!/usr/bin/env python3
"""Reproduce the stop-then-start race in ``TrainingLifecycleManager`` (report only; not fixed here).

Project: juniper-cascor
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-11-08
Status: ad-hoc — investigation: evidence for a reported race, which gets its own fix in its own
        PR by the owner's ruling 17 (2026-11-09, recorded in the record-update PR), "cascor:
        stop-then-start"
Retire when: that PR has merged
Related: juniper-ml notes/JUNIPER_2026-11-06_JUNIPER-ECOSYSTEM_CURSOR-FLOOD-4-DISPOSITION.md §8
         ruling 11 (the paused-fit wedge, whose fix PR reports this race); the same ruling lists
         juniper-service-core's stop-then-start race, which has its own shape

``stop_training`` raises the shared ``_stop_event`` and moves the FSM to STOPPED; the fit sees
the signal only at its next interrupt checkpoint. ``start_training`` accepts a start from
STOPPED and CLEARS that same event. A start sent after a stop but before the old fit reaches
its next checkpoint therefore cancels the stop: the old fit is never interrupted and trains on
to its natural end, while the FSM reads STOPPED and the new run waits queued behind it on the
single-worker executor. Every further start in that window is accepted too and queues another
run. When the old fit returns it still owns the lifecycle -- the new run has not entered
``_run_training`` -- so it settles: ``mark_completed`` is refused from STOPPED, yet
``training_state`` is written ``Completed`` and a successful session is counted, for a run
the operator stopped.

Unlike juniper-service-core, where the new run starts on its own thread at once and so already
owns the lifecycle when the old fit returns (juniper-ml#2531's thread check; the published 0.7.2
predates it, and there the old fit's return still marks the new run COMPLETED while it trains),
cascor's queue keeps the new run from starting until the old fit returns. The lifecycle-ownership
guard therefore does not change this race: the stale fit is still the owner when it settles.
``swap_dataset_live`` has a variant: when its 10 s join on the old future times out, it clears
``_stop_event`` before submitting the swapped run, which un-stops the old fit the same way.

Drives the real manager with a stub ``model.fit`` (the unit suite's house pattern), so it needs
no data service and spawns no candidate pool.

Usage:
    /opt/miniforge3/envs/JuniperCascor1/bin/python util/ad-hoc/2026-11-08_stop_then_start_race_repro.py [--src <cascor src dir>]
Exit: 1 when the race reproduces (the stop was cancelled), 0 when it does not.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WAIT = 5.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(ROOT / "src"), help="the cascor src/ tree to import (default: this checkout's)")
    args = parser.parse_args()
    sys.path.insert(0, args.src)

    import torch
    from juniper_model_core.events import TrainingEvent

    from api.lifecycle import manager as manager_module
    from api.lifecycle.manager import TrainingLifecycleManager

    print(f"manager under test: {manager_module.__file__}")

    entered = [threading.Event() for _ in range(3)]
    released = [threading.Event() for _ in range(3)]
    reached_checkpoint = [threading.Event() for _ in range(3)]
    calls = []

    def fit(x, y, *, X_val=None, y_val=None, on_event=None, **kwargs):
        index = len(calls)
        calls.append(index)
        on_event(TrainingEvent("training_start", {"n_samples": int(x.shape[0])}, 0))
        entered[index].set()
        released[index].wait(timeout=10.0)
        # The fit's next interrupt checkpoint. A stop that is still requested raises here.
        on_event(TrainingEvent("epoch_end", {"epoch": 25, "epochs": 50, "metrics": {"loss": 0.5}}, 1))
        reached_checkpoint[index].set()
        on_event(TrainingEvent("training_end", {"metrics": {}}, 2))

    mgr = TrainingLifecycleManager()
    mgr.create_network(input_size=2, output_size=2)
    mgr.model.fit = fit
    x = torch.randn(8, 2)
    y = torch.zeros(8, 2)
    y[:4, 0] = 1
    y[4:, 1] = 1

    def fsm() -> str:
        return mgr.state_machine.get_state_summary()["status"]

    # An ordered trail of what each run's settle did. The single worker starts run 1 the
    # moment run 0 returns, so reading the FSM afterwards would show run 1, not run 0's settle.
    trail = []
    sm = mgr.state_machine
    real_handle, real_mark_completed, real_update = sm.handle_command, sm.mark_completed, mgr.training_state.update_state
    real_counted = manager_module.inc_training_session_completed

    def handle_command(command):
        before = fsm()
        ok = real_handle(command)
        trail.append(f"FSM {command.name} from {before} -> {'accepted' if ok else 'refused'}")
        return ok

    def mark_completed():
        before = fsm()
        ok = real_mark_completed()
        trail.append(f"FSM mark_completed from {before} -> {'accepted' if ok else 'refused'}")
        return ok

    def update_state(**fields):
        if "status" in fields:
            trail.append(f"training_state.status := {fields['status']!r}")
        return real_update(**fields)

    def counted(status):
        trail.append(f"session counted as {status!r}")
        return real_counted(status)

    sm.handle_command, sm.mark_completed, mgr.training_state.update_state = handle_command, mark_completed, update_state
    manager_module.inc_training_session_completed = counted

    stop_cancelled = False
    try:
        mgr.start_training(X=x, y=y)
        assert entered[0].wait(WAIT), "run 0 never entered its fit"
        old_future = mgr._training_future
        old_future.add_done_callback(lambda _future: trail.append("--- run 0's _run_training returned ---"))
        print(f"run 0 training, between checkpoints: FSM {fsm()}")

        mgr.stop_training()
        print(f"stop requested: FSM {fsm()}, stop signal up: {mgr._stop_event.is_set()}")

        mgr.start_training(X=x, y=y)
        new_future = mgr._training_future
        print(f"start accepted before run 0's next checkpoint: FSM {fsm()}, stop signal up: {mgr._stop_event.is_set()}, run 1 queued: {not new_future.running() and not new_future.done()}")

        mgr.start_training(X=x, y=y)
        third_future = mgr._training_future
        print(f"a further start in the same window is accepted too: run 2 queued: {not third_future.running() and not third_future.done()}")

        released[0].set()
        old_future.result(timeout=WAIT)
        stop_cancelled = reached_checkpoint[0].is_set()
        print(f"run 0 passed its next checkpoint (the stop was cancelled): {stop_cancelled}")

        assert entered[1].wait(WAIT), "run 1 never started"
        released[1].set()
        new_future.result(timeout=WAIT)
        assert entered[2].wait(WAIT), "run 2 never started"
        released[2].set()
        third_future.result(timeout=WAIT)
        print(f"runs 1 and 2 then trained in turn: FSM {fsm()}, fits called {len(calls)}")
    finally:
        for event in released:
            event.set()
        mgr.shutdown()

    print("\nordered trail (run 0's settle is everything before its marker after the stop):")
    for entry in trail:
        print(f"  {entry}")

    print("RACE REPRODUCED: the stop was cancelled by the start" if stop_cancelled else "race not reproduced: the stop held")
    return 1 if stop_cancelled else 0


if __name__ == "__main__":
    sys.exit(main())
