#!/usr/bin/env python
"""
Test client for remote multiprocessing manager connection.
"""
import multiprocessing as mp
from multiprocessing.managers import BaseManager
# import time
# import torch
# import numpy as np
import sys
# import os

# Add the source directory to Python path
sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')

from candidate_unit.candidate_unit import CandidateUnit

class RemoteCandidateTrainingClient:
    def __init__(self, server_address=('127.0.0.1', 50000), authkey=b'Juniper_Cascade_Correlation_Multiprocessing_Authkey'):
        self.server_address = server_address
        self.authkey = authkey
        self.manager = None

    def connect(self):
        """Connect to the remote manager server."""
        try:
            # Define manager class
            class CandidateTrainingManager(BaseManager):
                pass

            # Register remote methods
            CandidateTrainingManager.register('get_tasks_queue')
            CandidateTrainingManager.register('get_done_queue')

            # Connect to remote manager
            self.manager = CandidateTrainingManager(
                address=self.server_address,
                authkey=self.authkey
            )

            self.manager.connect()
            print(f"Successfully connected to manager at {self.server_address}")
            return True

        except Exception as e:
            print(f"Failed to connect to manager: {e}")
            return False

    def process_tasks(self, num_workers=2):  # sourcery skip: extract-method
        """Process tasks from the remote queue."""
        if not self.manager:
            print("Not connected to manager")
            return

        try:
            tasks_queue = self.manager.get_tasks_queue()
            done_queue = self.manager.get_done_queue()
            print(f"Starting {num_workers} worker processes")

            # Start worker processes
            processes = []
            for i in range(num_workers):
                p = mp.Process(target=self._worker_process, args=(tasks_queue, done_queue, i))
                p.start()
                processes.append(p)

            # Wait for processes to complete
            for p in processes:
                p.join()

            print("All worker processes completed")

        except Exception as e:
            print(f"Error processing tasks: {e}")

    @staticmethod
    def _worker_process(tasks_queue, done_queue, worker_id):
        """Worker process to handle tasks."""
        print(f"Worker {worker_id} started")

        processed_count = 0

        while True:
            try:
                # Get task with timeout
                task = tasks_queue.get(timeout=10)

                if task is None:  # Sentinel to stop
                    break

                print(f"Worker {worker_id} processing task {task[0]}")

                # Process the task
                result = RemoteCandidateTrainingClient._train_candidate_remote(task)

                # Put result back
                done_queue.put(result)
                processed_count += 1
                print(f"Worker {worker_id} completed task {task[0]} (total: {processed_count})")

            except Exception as e:
                if "timed out" not in str(e).lower():
                    print(f"Worker {worker_id} error: {e}")
                break

        print(f"Worker {worker_id} finished, processed {processed_count} tasks")

    @staticmethod
    def _train_candidate_remote(task_data):
        """Train a candidate unit remotely."""
        try:
            candidate_index, candidate_data, training_inputs = task_data
            # Unpack candidate data  
            (_, input_size, activation_name, random_value_scale, candidate_uuid, candidate_seed, random_max_value, sequence_max_value) = candidate_data
            # Unpack training inputs
            (candidate_input, candidate_epochs, y, residual_error, candidate_learning_rate, candidate_display_frequency) = training_inputs
            # Create and train candidate
            candidate = CandidateUnit(
                CandidateUnit__input_size=input_size,
                CandidateUnit__random_seed=candidate_seed,
                CandidateUnit__random_value_scale=random_value_scale,
                CandidateUnit__uuid=candidate_uuid,
                CandidateUnit__epochs=candidate_epochs,
                CandidateUnit__learning_rate=candidate_learning_rate,
                CandidateUnit__display_frequency=candidate_display_frequency,
            )

            # Train the candidate
            correlation, _ = candidate.train(
                x=candidate_input,
                epochs=candidate_epochs,
                residual_error=residual_error,
                learning_rate=candidate_learning_rate,
                display_frequency=candidate_display_frequency
            )

            return (candidate_index, candidate_uuid, correlation, candidate)

        except Exception as e:
            print(f"Remote training error: {e}")
            return (candidate_index if 'candidate_index' in locals() else None, candidate_uuid if 'candidate_uuid' in locals() else None, 0.0, None)

def test_remote_connection():
    """Test the remote connection functionality."""
    print("Testing remote multiprocessing manager connection...")

    # Create client
    client = RemoteCandidateTrainingClient()

    # Try to connect
    if client.connect():
        print("Connection successful!")

        # Process some tasks
        client.process_tasks(num_workers=2)

    else:
        print("Connection failed!")

if __name__ == "__main__":
    test_remote_connection()
