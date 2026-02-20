"""Training lifecycle manager — central coordinator for CasCor training.

Wraps CascadeCorrelationNetwork with:
- Thread-safe training via ThreadPoolExecutor
- State machine for deterministic control flow
- Monitoring hooks for real-time metrics
- Topology and statistics extraction
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import torch

from api.lifecycle.monitor import TrainingMonitor, TrainingState
from api.lifecycle.state_machine import Command, TrainingStateMachine


class TrainingLifecycleManager:
    """Central coordinator for CasCor network training lifecycle.

    Manages network creation, training execution (async via ThreadPoolExecutor),
    monitoring hooks, state tracking, and metrics collection.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core components
        self.network = None
        self.state_machine = TrainingStateMachine()
        self.training_state = TrainingState()
        self.training_monitor = TrainingMonitor()

        # Threading
        self._training_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._topology_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._training_future: Optional[Future] = None
        self._stop_requested = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # Monkey-patched originals
        self._original_methods: Dict[str, Callable] = {}
        self._monitoring_active = False

        # Training data
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._val_x: Optional[torch.Tensor] = None
        self._val_y: Optional[torch.Tensor] = None

        # Network creation params (for reset)
        self._network_params: Optional[Dict[str, Any]] = None

        self.logger.info("TrainingLifecycleManager initialized")

    # ------------------------------------------------------------------
    # Network management
    # ------------------------------------------------------------------

    def create_network(self, **kwargs) -> Dict[str, Any]:
        """Create a new CascadeCorrelationNetwork.

        Args:
            **kwargs: Parameters passed to CascadeCorrelationConfig.create_simple_config()

        Returns:
            Network info dictionary
        """
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Cannot create network while training is active")

            self._network_params = kwargs.copy()
            config = CascadeCorrelationConfig.create_simple_config(**kwargs)
            self.network = CascadeCorrelationNetwork(config=config)
            self._install_monitoring_hooks()

            self.training_state.update_state(
                status="Stopped",
                phase="Idle",
                learning_rate=kwargs.get("learning_rate", 0.01),
                max_hidden_units=kwargs.get("max_hidden_units", 10),
                max_epochs=kwargs.get("epochs_max", 200),
                network_name=f"CasCor-{kwargs.get('input_size', 2)}x{kwargs.get('output_size', 2)}",
            )

            info = self.get_network_info()
            self.logger.info(f"Network created: {info['input_size']}x{info['output_size']}")
            return info

    def delete_network(self) -> None:
        """Delete the current network."""
        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Cannot delete network while training is active")
            self._restore_original_methods()
            self.network = None
            self._network_params = None
            self.state_machine.handle_command(Command.RESET)
            self.training_state.update_state(status="Stopped", phase="Idle")
            self.logger.info("Network deleted")

    def has_network(self) -> bool:
        return self.network is not None

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        if self.network is None:
            return {}
        return {
            "input_size": self.network.input_size,
            "output_size": self.network.output_size,
            "hidden_units": len(self.network.hidden_units),
            "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
            "learning_rate": getattr(self.network, "learning_rate", 0.0),
            "uuid": str(getattr(self.network, "uuid", "")),
        }

    # ------------------------------------------------------------------
    # Monitoring hooks (monkey-patch approach from CascorIntegration)
    # ------------------------------------------------------------------

    def _install_monitoring_hooks(self) -> None:
        """Install monitoring hooks on the network via monkey-patching."""
        if self.network is None or self._monitoring_active:
            return

        original_fit = self.network.fit
        self._original_methods["fit"] = original_fit

        monitor = self.training_monitor
        state = self.training_state
        stop_event = self._stop_requested
        sm = self.state_machine
        manager_ref = self

        def monitored_fit(x, y, x_val=None, y_val=None, **kwargs):
            monitor.on_training_start()
            sm.handle_command(Command.START)
            state.update_state(status="Started", phase="Output")

            try:
                result = original_fit(x, y, x_val=x_val, y_val=y_val, **kwargs)

                # Extract metrics after fit completes
                manager_ref._extract_and_record_metrics()

                if stop_event.is_set():
                    sm.handle_command(Command.STOP)
                    state.update_state(status="Stopped", phase="Idle")
                else:
                    sm.mark_completed()
                    state.update_state(status="Completed", phase="Idle")

                return result
            except Exception as e:
                sm.mark_failed(str(e))
                state.update_state(status="Failed", phase="Idle")
                raise
            finally:
                monitor.on_training_end()

        self.network.fit = monitored_fit
        self._monitoring_active = True
        self.logger.info("Monitoring hooks installed")

    def _restore_original_methods(self) -> None:
        """Restore original network methods."""
        if not self._original_methods or self.network is None:
            return
        for method_name, original in self._original_methods.items():
            setattr(self.network, method_name, original)
        self._original_methods.clear()
        self._monitoring_active = False
        self.logger.info("Original methods restored")

    def _extract_and_record_metrics(self) -> None:
        """Extract current metrics from network history and record them."""
        if self.network is None or not hasattr(self.network, "history"):
            return
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss_list = list(history.get("train_loss", []))
                train_accuracy_list = list(history.get("train_accuracy", []))
                val_loss_list = list(history.get("value_loss", []))
                val_accuracy_list = list(history.get("value_accuracy", []))
                hidden_units_count = len(self.network.hidden_units)
            except (RuntimeError, KeyError):
                return

        epoch = len(train_loss_list)
        if epoch > 0:
            self.training_monitor.on_epoch_end(
                epoch=epoch,
                loss=train_loss_list[-1] if train_loss_list else 0.0,
                accuracy=train_accuracy_list[-1] if train_accuracy_list else 0.0,
                learning_rate=getattr(self.network, "learning_rate", 0.0),
                hidden_units=hidden_units_count,
                validation_loss=val_loss_list[-1] if val_loss_list else None,
                validation_accuracy=val_accuracy_list[-1] if val_accuracy_list else None,
            )
            self.training_state.update_state(
                current_epoch=epoch,
                current_step=epoch,
            )

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Start training asynchronously.

        Args:
            x: Training features tensor
            y: Training targets tensor
            x_val: Validation features
            y_val: Validation targets
            **kwargs: Additional kwargs passed to network.fit()

        Returns:
            Status dictionary
        """
        if self.network is None:
            raise RuntimeError("No network created")

        with self._training_lock:
            if self.state_machine.is_started():
                raise RuntimeError("Training already in progress")

            if x is not None:
                self._train_x = x
                self._train_y = y
            if x_val is not None:
                self._val_x = x_val
                self._val_y = y_val

            if self._train_x is None or self._train_y is None:
                raise ValueError("Training data not provided")

            self._stop_requested.clear()
            self._pause_event.set()

            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascor-train")

            self._training_future = self._executor.submit(self._run_training, self._train_x, self._train_y, self._val_x, self._val_y, **kwargs)

        return {"status": "training_started", "timestamp": time.time()}

    def _run_training(self, x, y, x_val, y_val, **kwargs) -> None:
        """Execute training in background thread."""
        try:
            self.network.fit(x, y, x_val=x_val, y_val=y_val, **kwargs)
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)

    def stop_training(self) -> Dict[str, Any]:
        """Request training stop."""
        self._stop_requested.set()
        self.state_machine.handle_command(Command.STOP)
        self.training_state.update_state(status="Stopped", phase="Idle")
        return {"status": "stop_requested", "timestamp": time.time()}

    def pause_training(self) -> Dict[str, Any]:
        """Pause training."""
        if not self.state_machine.is_started():
            raise RuntimeError("Training is not active")
        self._pause_event.clear()
        self.state_machine.handle_command(Command.PAUSE)
        self.training_state.update_state(status="Paused")
        return {"status": "paused", "timestamp": time.time()}

    def resume_training(self) -> Dict[str, Any]:
        """Resume paused training."""
        if not self.state_machine.is_paused():
            raise RuntimeError("Training is not paused")
        self._pause_event.set()
        self.state_machine.handle_command(Command.RESUME)
        self.training_state.update_state(status="Started")
        return {"status": "resumed", "timestamp": time.time()}

    def reset(self) -> Dict[str, Any]:
        """Reset training state."""
        self._stop_requested.set()
        self.state_machine.handle_command(Command.RESET)
        self.training_monitor.clear_metrics()
        self.training_state.update_state(
            status="Stopped",
            phase="Idle",
            current_epoch=0,
            current_step=0,
        )
        return {"status": "reset", "timestamp": time.time()}

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        state_summary = self.state_machine.get_state_summary()
        monitor_state = self.training_monitor.get_current_state()
        training_state = self.training_state.get_state()

        return {
            "state_machine": state_summary,
            "monitor": monitor_state,
            "training_state": training_state,
            "network_loaded": self.network is not None,
            "training_active": self.state_machine.is_started(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if self.network is None:
            return {}
        with self._metrics_lock:
            try:
                history = self.network.history
                train_loss = list(history.get("train_loss", []))
                train_accuracy = list(history.get("train_accuracy", []))
                val_loss = list(history.get("value_loss", []))
                val_accuracy = list(history.get("value_accuracy", []))
                hidden_units = len(self.network.hidden_units)
            except (RuntimeError, KeyError):
                return {}

        return {
            "epoch": len(train_loss),
            "train_loss": train_loss[-1] if train_loss else None,
            "train_accuracy": train_accuracy[-1] if train_accuracy else None,
            "val_loss": val_loss[-1] if val_loss else None,
            "val_accuracy": val_accuracy[-1] if val_accuracy else None,
            "hidden_units": hidden_units,
            "timestamp": datetime.now().isoformat(),
        }

    def get_metrics_history(self, count: Optional[int] = None) -> list:
        """Get metrics history."""
        if count:
            return self.training_monitor.get_recent_metrics(count)
        return self.training_monitor.get_all_metrics()

    def get_training_params(self) -> Dict[str, Any]:
        """Get current training parameters."""
        if self.network is None:
            return {}
        return {
            "learning_rate": getattr(self.network, "learning_rate", 0.0),
            "max_hidden_units": getattr(self.network, "max_hidden_units", 0),
            "epochs_max": getattr(self.network, "epochs_max", 0),
            "patience": getattr(self.network, "patience", 0),
            "candidate_pool_size": getattr(self.network, "candidate_pool_size", 0),
            "correlation_threshold": getattr(self.network, "correlation_threshold", 0.0),
        }

    # ------------------------------------------------------------------
    # Topology & statistics
    # ------------------------------------------------------------------

    def get_topology(self) -> Optional[Dict[str, Any]]:
        """Extract network topology for visualization (thread-safe)."""
        if self.network is None:
            return None
        try:
            with self._topology_lock:
                with torch.no_grad():
                    topology = {
                        "input_size": self.network.input_size,
                        "output_size": self.network.output_size,
                        "hidden_units": [],
                        "output_weights": self.network.output_weights.detach().cpu().tolist(),
                        "output_bias": self.network.output_bias.detach().cpu().tolist(),
                    }
                    for i, unit in enumerate(self.network.hidden_units):
                        topology["hidden_units"].append(
                            {
                                "id": i,
                                "weights": unit["weights"].detach().cpu().tolist(),
                                "bias": float(unit["bias"]),
                                "activation": unit.get("activation_fn", torch.sigmoid).__name__,
                            }
                        )
            return topology
        except Exception as e:
            self.logger.error(f"Failed to extract topology: {e}", exc_info=True)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get network weight statistics."""
        if self.network is None:
            return {}
        try:
            with self._topology_lock:
                with torch.no_grad():
                    output_weights = self.network.output_weights.detach().cpu()
                    stats = {
                        "total_hidden_units": len(self.network.hidden_units),
                        "output_weight_mean": float(output_weights.mean()),
                        "output_weight_std": float(output_weights.std()),
                        "output_weight_min": float(output_weights.min()),
                        "output_weight_max": float(output_weights.max()),
                    }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean up resources."""
        self._stop_requested.set()
        self._restore_original_methods()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self.logger.info("TrainingLifecycleManager shut down")
