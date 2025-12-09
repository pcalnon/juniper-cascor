from typing import Dict, List

import BaseManager
import CandidateTrainingManager
import CandidateUnit
import torch


def _create_multiprocessing_manager(
    self, candidate_training_context=None, start_manager=True
) -> BaseManager:
    """
    Create and configure the multiprocessing manager for remote access.
    """
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Creating multiprocessing manager for candidate training."
    )

    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: candidate_training_context provided: {candidate_training_context is not None}"
    )
    candidate_training_context = (
        candidate_training_context or self.get_candidate_training_context()
    )
    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Using candidate_training_context: {candidate_training_context}"
    )
    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: start_manager flag: {start_manager}"
    )

    # Compile list of modules to be pre-loaded into the forkserver context
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Compiling list of modules to be pre-loaded into the forkserver context."
    )
    logging_modules_list = ["logging", "logging.config", "datetime"]
    torch_modules_list = ["torch", "torch.nn", "torch.optim"]
    multiprocessing_modules_list = [
        "multiprocessing.current_process",
        "multiprocessing.managers.BaseManager",
    ]
    os_modules_list = [
        "os",
        "sys",
        "numpy",
        "random",
        "math.inf",
        "uuid",
        "traceback",
        "typing.Optional",
        "typing.Dict",
        "typing.List",
    ]
    candidate_unit_modules_list = [
        "candidate_unit.candidate_unit.CandidateUnit",
        "constants.constants",
        "utils.utils.display_progress",
        "log_config.log_config.LogConfig",
        "log_config.logger.logger.Logger",
    ]

    # Preload modules into the candidate_training_context
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Preloading modules into the candidate_training_context"
    )
    candidate_training_context_modules = (
        logging_modules_list
        + torch_modules_list
        + multiprocessing_modules_list
        + os_modules_list
        + candidate_unit_modules_list
    )
    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Preloading modules into candidate training context: {candidate_training_context_modules}"
    )

    try:
        candidate_training_context.set_forkserver_preload(
            candidate_training_context_modules
        )
        self.logger.debug(
            f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Preloaded modules into candidate training context: {candidate_training_context_modules}"
        )
    except Exception as e:
        self.logger.warning(
            f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Failed to preload modules: {e}"
        )

    # Create shared queues using the context
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Creating shared queues"
    )
    candidate_training_tasks_queue = candidate_training_context.Queue()
    candidate_training_done_queue = candidate_training_context.Queue()

    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Created candidate_training_tasks_queue: Type: {type(candidate_training_tasks_queue)}"
    )
    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Created candidate_training_done_queue: Type: {type(candidate_training_done_queue)}"
    )

    # Register queue methods with the manager class
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Registering queue methods with CandidateTrainingManager"
    )
    CandidateTrainingManager.register(
        "get_tasks_queue", callable=lambda: candidate_training_tasks_queue
    )
    CandidateTrainingManager.register(
        "get_done_queue", callable=lambda: candidate_training_done_queue
    )
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Registered queue methods with CandidateTrainingManager"
    )

    self.logger.debug(
        f"address: type: {type(self.candidate_training_queue_address)}, value: {self.candidate_training_queue_address}"
    )
    self.logger.debug(
        f"authkey: type: {type(self.candidate_training_queue_authkey)}, value: {self.candidate_training_queue_authkey}"
    )
    self.logger.debug(
        f"context: type: {type(candidate_training_context)}, value: {candidate_training_context}"
    )

    # Create the manager instance
    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Creating manager instance"
    )
    manager = CandidateTrainingManager(
        address=self.candidate_training_queue_address,
        authkey=self.candidate_training_queue_authkey,
        ctx=candidate_training_context,
    )
    self.logger.debug(
        f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Created manager: type: {type(manager)}"
    )

    # Start the manager server if requested
    if start_manager:
        self.logger.debug(
            "CascadeCorrelationNetwork: _create_multiprocessing_manager: Starting manager server"
        )
        try:
            manager.start()
            self.logger.info(
                f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Manager server started at {self.candidate_training_queue_address}"
            )

            # Give the server a moment to fully initialize
            import time

            time.sleep(0.1)

        except Exception as e:
            self.logger.error(
                f"CascadeCorrelationNetwork: _create_multiprocessing_manager: Failed to start manager server: {e}"
            )
            raise

    self.logger.debug(
        "CascadeCorrelationNetwork: _create_multiprocessing_manager: Completed creation of Manager Server."
    )
    return manager


def start_manager_server_process(self, manager):
    """
    Start the manager server in a separate process for remote access.
    """
    self.logger.debug(
        "CascadeCorrelationNetwork: start_manager_server_process: Starting manager server process"
    )

    def server_process_target():
        """Target function for the server process"""
        try:
            self.logger.info(
                "CascadeCorrelationNetwork: Manager server process started, serving forever..."
            )
            server = manager.get_server()
            server.serve_forever()
        except Exception as e:
            self.logger.error(
                f"CascadeCorrelationNetwork: Manager server process error: {e}"
            )

    # Create and start the server process
    server_process = self.candidate_training_context.Process(
        target=server_process_target, name="CandidateTrainingManagerServer"
    )
    server_process.daemon = True  # Dies when parent dies
    server_process.start()

    self.logger.info(
        f"CascadeCorrelationNetwork: Manager server process started with PID: {server_process.pid}"
    )
    return server_process


def train_candidates(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    residual_error: torch.Tensor,
    tasks: List[Dict],
) -> List[CandidateUnit]:
    """
    Train a pool of candidate units based on the residual error from the network, and select the best one.
    Fixed version with proper manager server handling.
    """
    # ... [existing code for preparation] ...

    # Create and start the multiprocessing manager for remote access
    candidate_training_manager = None
    server_process = None
    results = []

    try:
        # Create the manager (but don't start it yet)
        candidate_training_manager = self._create_multiprocessing_manager(
            start_manager=False
        )

        # Start the manager server
        candidate_training_manager.start()
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Manager server started at {self.candidate_training_queue_address}"
        )

        # Give the server time to fully start
        import time

        time.sleep(0.2)

        # Now we can safely get the queues
        candidate_training_shared_tasks_queue = (
            candidate_training_manager.get_tasks_queue()
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: train_candidates: Retrieved tasks queue: {candidate_training_shared_tasks_queue}"
        )

        candidate_training_shared_done_queue = (
            candidate_training_manager.get_done_queue()
        )
        self.logger.verbose(
            f"CascadeCorrelationNetwork: train_candidates: Retrieved done queue: {candidate_training_shared_done_queue}"
        )

        # Add tasks to the queue
        for task in tasks:
            candidate_training_shared_tasks_queue.put(task)
        self.logger.debug(
            f"CascadeCorrelationNetwork: train_candidates: Added {len(tasks)} tasks to queue"
        )

        # Process tasks with worker processes or collect from remote workers
        if self.candidate_training_processes > 1:
            with self.candidate_training_context.Pool(
                processes=self.candidate_training_processes
            ) as candidate_training_pool:
                self.logger.debug(
                    f"CascadeCorrelationNetwork: train_candidates: Created pool with {self.candidate_training_processes} processes"
                )

                # Create worker arguments for local processing
                worker_args = [
                    (
                        candidate_training_shared_tasks_queue,
                        candidate_training_shared_done_queue,
                        self.candidate_training_queue_address,
                        self.candidate_training_queue_authkey,
                        self.candidate_training_tasks_queue_timeout,
                        True,
                    )
                    for _ in range(self.candidate_training_processes)
                ]

                # Start local workers
                candidate_training_pool.starmap(self._worker_process, worker_args)
                candidate_training_pool.close()
                candidate_training_pool.join()

        # Collect results from the done queue
        collected_results = 0
        timeout_counter = 0
        max_timeout = 30

        while collected_results < len(tasks) and timeout_counter < max_timeout:
            try:
                if not candidate_training_shared_done_queue.empty():
                    result = candidate_training_shared_done_queue.get(timeout=1)
                    results.append(result)
                    collected_results += 1
                    timeout_counter = 0
                else:
                    time.sleep(0.1)
                    timeout_counter += 0.1
            except Exception as e:
                self.logger.warning(f"Error collecting results: {e}")
                timeout_counter += 1

        self.logger.info(
            f"CascadeCorrelationNetwork: train_candidates: Collected {len(results)} results"
        )

    except Exception as e:
        self.logger.error(
            f"CascadeCorrelationNetwork: train_candidates: Manager error: {e}"
        )
        # Fallback to sequential processing
        self.logger.info(
            "CascadeCorrelationNetwork: train_candidates: Falling back to sequential processing"
        )
        results = []
        for task in tasks:
            try:
                result = self.train_candidate_worker(task, parallel=False)
                results.append(result)
            except Exception as task_e:
                self.logger.error(f"Task error: {task_e}")
                results.append(
                    (task[0], task[1][4] if len(task[1]) > 4 else None, 0.0, None)
                )

    finally:
        # Clean up manager and server process
        if candidate_training_manager:
            try:
                self.logger.debug(
                    "CascadeCorrelationNetwork: train_candidates: Shutting down manager"
                )
                candidate_training_manager.shutdown()
                self.logger.debug(
                    "CascadeCorrelationNetwork: train_candidates: Manager shutdown completed"
                )
            except Exception as cleanup_e:
                self.logger.warning(f"Manager cleanup warning: {cleanup_e}")

        if server_process and server_process.is_alive():
            try:
                server_process.terminate()
                server_process.join(timeout=5)
                self.logger.debug(
                    "CascadeCorrelationNetwork: train_candidates: Server process terminated"
                )
            except Exception as cleanup_e:
                self.logger.warning(f"Server process cleanup warning: {cleanup_e}")

    # ... [rest of existing result processing code] ...
