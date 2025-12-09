#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     remote_client.py
# Author:        Paul Calnon
# Version:       0.3.2 (0.7.3)
#
# Date:          2025-09-27
# Last Modified: 2025-09-27
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    Remote worker client for connecting to CascadeCorrelationNetwork multiprocessing servers.
#    Provides secure remote access to shared queues and manages local worker processes.
#
#####################################################################################################################################################################################################

import multiprocessing as mp
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from log_config.logger.logger import Logger


class RemoteWorkerClient:
    """
    Client class for connecting to remote CandidateTrainingManager servers.
    Provides secure remote access to shared queues and manages local worker processes.
    """

    def __init__(self, address, authkey, ctx=None, logger=None):
        """
        Initialize remote worker client.
        Args:
            address: Tuple of (host, port) for the remote manager
            authkey: Authentication key (string or bytes)
            ctx: Multiprocessing context (defaults to forkserver)
            logger: Logger instance
        """
        self.ctx = ctx or mp.get_context('forkserver')
        self.logger = logger or Logger

        if isinstance(authkey, str):
            authkey = authkey.encode('utf-8')

        self.address = address
        self.authkey = authkey
        self.manager = None
        self.task_queue = None
        self.result_queue = None
        self.workers = []

        self.logger.debug(f"RemoteWorkerClient: __init__: Initialized client for {address}")

    def connect(self):
        """Connect to the remote manager server."""
        try:
            # Import here to avoid circular import
            from cascade_correlation.cascade_correlation import CandidateTrainingManager

            self.manager = CandidateTrainingManager(address=self.address, authkey=self.authkey)
            self.manager.connect()

            # Obtain queue proxies
            self.task_queue = self.manager.get_task_queue()
            self.result_queue = self.manager.get_result_queue()

            self.logger.info(f"RemoteWorkerClient: connect: Connected to remote manager at {self.address}")
        except Exception as e:
            self.logger.error(f"RemoteWorkerClient: connect: Failed to connect to remote manager: {e}")
            raise

    def start_workers(self, num_workers=1):
        """
        Start local worker processes that consume from remote queues.
        Args:
            num_workers: Number of worker processes to start
        """
        if not self.manager:
            raise RuntimeError("Must call connect() before starting workers")

        self.logger.debug(f"RemoteWorkerClient: start_workers: Starting {num_workers} worker processes")

        for i in range(num_workers):
            # Import here to avoid circular import
            from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork

            worker = self.ctx.Process(
                target=CascadeCorrelationNetwork._worker_loop,
                args=(self.task_queue, self.result_queue, True),
                daemon=True,
                name=f"RemoteWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)

        self.logger.info(f"RemoteWorkerClient: start_workers: Started {len(self.workers)} worker processes")

    def stop_workers(self, timeout=10):
        """
        Stop all worker processes by sending sentinel values.
        Args:
            timeout: Timeout in seconds to wait for workers to stop
        """
        if not self.workers:
            return

        self.logger.debug(f"RemoteWorkerClient: stop_workers: Stopping {len(self.workers)} worker processes")

        # Send sentinel None for each worker
        for _ in self.workers:
            try:
                self.task_queue.put(None)
            except Exception as e:
                self.logger.error(f"RemoteWorkerClient: stop_workers: Failed to send sentinel: {e}")

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                self.logger.warning(f"RemoteWorkerClient: stop_workers: Worker {worker.name} did not stop gracefully, terminating")
                worker.terminate()

        self.workers.clear()
        self.logger.info("RemoteWorkerClient: stop_workers: All worker processes stopped")

    def disconnect(self):
        """Disconnect from the remote manager."""
        if self.workers:
            self.stop_workers()

        if self.manager:
            try:
                # Note: Client connections don't need explicit disconnect, but we clean up references
                self.manager = None
                self.task_queue = None
                self.result_queue = None
                self.logger.info("RemoteWorkerClient: disconnect: Disconnected from remote manager")
            except Exception as e:
                self.logger.error(f"RemoteWorkerClient: disconnect: Error during disconnect: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
