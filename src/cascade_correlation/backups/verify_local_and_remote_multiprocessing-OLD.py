#!/usr/bin/env python
"""
Comprehensive test for both local and remote multiprocessing.
"""
import multiprocessing as mp

# import pytest
import platform
import sys

# import os
import time
from multiprocessing.managers import BaseManager

import numpy as np
import torch

# from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
from cascade_correlation import CascadeCorrelationNetwork
from constants.constants import (
    _PROJECT_MODEL_AUTHKEY,  # Define base manager server authkey for remote multiprocessing shared queues: 'Juniper_Cascade_Correlation_Multiprocessing_Authkey'
)
from constants.constants import (
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS,  # Define base manager server address for remote multiprocessing shared queues
)
from constants.constants import (  # _PROJECT_MODEL_BASE_MANAGER_HOSTNAME,      # Define base manager server hostname for remote multiprocessing shared queues = 'localhost',; _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP,    # Define base manager server IP address for remote multiprocessing shared queues = '127.0.0.1',; _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT,  # Define base manager server port for remote multiprocessing shared queues = 50000; _PROJECT_MODEL_SHUTDOWN_TIMEOUT,; _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
    _PROJECT_TESTING_FAILED_TEST,
    _PROJECT_TESTING_PARTIAL_TEST,
    _PROJECT_TESTING_PASSED_TEST,
    _PROJECT_TESTING_SKIPPED_TEST,
    _PROJECT_TESTING_UNKNOWN_TEST,
    _PROJECT_TESTING_UNSTABLE_TEST,
)

_CANDIDATE_POOL_SIZE = 20
_CANDIDATE_EPOCHS = (5,)

# Add the source directory to Python path
if platform.system() == "Linux":
    sys.path.append(
        "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src"
    )
elif platform.system() == "Darwin":  # macOS
    sys.path.append(
        "/Users/pcalnon/Development/python/Juniper/src/prototypes/cascor/src"
    )
else:
    raise ValueError("Unsupported operating system.")


network = None  # Global network instance for remote testing
manager_context = None  # Global manager context for remote testing


############################################################################################################################################################
def create_test_data():
    """Create test data for spiral problem."""
    # Create simple 2D spiral data
    n_points = 50
    t = torch.linspace(0, 2 * np.pi, n_points)

    # Create two spirals
    spiral1_x = t * torch.cos(t) * 0.1
    spiral1_y = t * torch.sin(t) * 0.1
    spiral2_x = -t * torch.cos(t) * 0.1
    spiral2_y = -t * torch.sin(t) * 0.1

    # Combine spirals
    x = torch.cat(
        [
            torch.stack([spiral1_x, spiral1_y], dim=1),
            torch.stack([spiral2_x, spiral2_y], dim=1),
        ]
    )
    # Create labels (0 for first spiral, 1 for second)
    y = torch.cat([torch.zeros(n_points, 1), torch.ones(n_points, 1)])

    return x, y


############################################################################################################################################################
def test_local_multiprocessing():  # sourcery skip: extract-method
    """Test local multiprocessing functionality."""
    print("\n=== Testing Local Multiprocessing ===")

    try:
        # Create network with smaller candidate pool for testing
        network = CascadeCorrelationNetwork(
            _CascadeCorrelationNetwork__candidate_pool_size=4,
            _CascadeCorrelationNetwork__candidate_epochs=5,
        )

        # Create test data
        x, y = create_test_data()
        print(f"Test data shapes: x={x.shape}, y={y.shape}")

        # Calculate initial residual error
        residual_error = network.calculate_residual_error(x, y)
        print(f"Residual error shape: {residual_error.shape}")

        # Train candidates
        print("Starting candidate training...")
        start_time = time.time()
        results = network.train_candidates(x, y, residual_error)
        end_time = time.time()

        # extract results
        candidates_list, best_candidate, max_correlation = results
        candidate_ids, candidate_uuids, correlations, candidates = candidates_list

        print(
            f"\n{_PROJECT_TESTING_PARTIAL_TEST} Training completed in {end_time - start_time:.2f} seconds"
        )
        print(
            f"\n{_PROJECT_TESTING_PARTIAL_TEST} Total candidates trained: {len(candidate_ids)}"
        )
        print(
            f"\n{_PROJECT_TESTING_PARTIAL_TEST} Best correlation: {max_correlation[0]:.6f}"
        )
        print(
            f"\n{_PROJECT_TESTING_PARTIAL_TEST} Successful candidates: {max_correlation[1]}"
        )
        print(
            f"\n{_PROJECT_TESTING_PARTIAL_TEST} Failed candidates: {max_correlation[2]}"
        )
        return len(candidate_ids) > 0 and max_correlation[1] > 0

    except Exception as e:
        print(
            f"\n{_PROJECT_TESTING_FAILED_TEST} Local multiprocessing test failed: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


############################################################################################################################################################
def initialize_network_and_context(
    candidate_pool_size=_CANDIDATE_POOL_SIZE,
    candidate_epochs=_CANDIDATE_EPOCHS,
):  # sourcery skip: extract-duplicate-method
    """Initialize network and manager context."""
    print("\n=== Initializing Network and Manager Context ===")
    global network, manager_context
    print("Creating CascadeCorrelationNetwork object...")
    network = CascadeCorrelationNetwork(
        _CascadeCorrelationNetwork__candidate_pool_size=candidate_pool_size,
        _CascadeCorrelationNetwork__candidate_epochs=candidate_epochs,
    )
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created CascadeCorrelationNetwork object..."
    )
    print(f"Candidate training context: type: {type(network)}, value: {network}")
    print("Getting candidate training context from network...")
    manager_context = network.get_candidate_training_context()
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Retrieved candidate training context from network..."
    )
    print(
        f"Candidate training context: type: {type(manager_context)}, value: {manager_context}"
    )
    print("Completed network and manager context initialization.")
    return network, manager_context


def initialize_manager():
    """Initialize multiprocessing manager."""
    print("\n=== Initializing Multiprocessing Manager ===")
    global network, manager_context  # Use the global network instance for remote testing

    print("Accessing candidate pool size and epochs from network...")
    candidate_pool_size = network.get_candidate_pool_size()
    candidate_epochs = network.get_candidate_epochs()
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully accessed candidate pool size and epochs from network..."
    )
    print(
        f"Candidate pool size: {candidate_pool_size}, candidate epochs: {candidate_epochs}"
    )

    print("Accessing candidate training queue address from network...")
    candidate_training_queue_address = network.get_candidate_training_queue_address()
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully accessed candidate training queue address from network..."
    )
    print(f"Candidate training queue address: {candidate_training_queue_address}")

    # Create test data
    print("Creating test data...")
    x, y = create_test_data()
    print(f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created test data...")
    print(f"Test data shapes: x={x.shape}, y={y.shape}")

    # Calculate initial residual error
    print("Calculating initial residual error...")
    residual_error = network.calculate_residual_error(x, y)
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully calculated initial residual error..."
    )
    print(f"Residual error shape: {residual_error.shape}")

    # Create manager
    print("Creating multiprocessing manager...")
    manager = network._create_multiprocessing_manager(
        candidate_training_context=manager_context, start_manager=False
    )
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created multiprocessing manager..."
    )
    print(f"Manager details: type: {type(manager)}, value: {manager}")

    print("Accessing candidate training context from manager...")
    manager_context = network.get_candidate_training_context()
    print(
        f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully accessed candidate training context from manager..."
    )
    print(
        f"Candidate training context: type: {type(manager_context)}, value: {manager_context}"
    )

    return (
        manager,
        x,
        y,
        residual_error,
        candidate_pool_size,
        candidate_epochs,
        candidate_training_queue_address,
    )


############################################################################################################################################################
def start_manager_server(
    manager: BaseManager = None,
):  # sourcery skip: extract-duplicate-method, extract-method
    """Start a manager server for testing."""
    print("\n=== Starting Manager Server ===")

    try:
        print("Creating Manager Server...")
        candidate_training_server = manager.get_server()
        print(f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created manager server...")
        print(
            f"Manager server details: type: {type(candidate_training_server)}, value: {candidate_training_server}"
        )
        print("Starting manager server...")
        candidate_training_server.serve_forever()
        print(
            f"{_PROJECT_TESTING_PARTIAL_TEST} Manager server has been started and is serving forever."
        )
        candidate_training_server.start()
        print(f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully started manager server...")
        print(
            f"Manager server details after start: type: {type(candidate_training_server)}, value: {candidate_training_server}"
        )
        time.sleep(120)  # run server for a while
        print("Returning manager server...")
        return (candidate_training_server,)

    except Exception as e:
        print(f"\n{_PROJECT_TESTING_FAILED_TEST} Failed to start manager server: {e}")
        return (candidate_training_server,)


############################################################################################################################################################
# NOTE: the process launched in this function is for client process that connects to the server manager to test the remote connection
def test_manager_connection():  # sourcery skip: extract-method
    """Test connection to manager server."""
    print("\n=== Testing Manager Connection ===")

    global network, manager_context  # Use the global network instance for remote testing

    try:
        remote_context = mp.get_context(method="forkserver")
        # Define manager class
        print("Defining manager class...")

        class CandidateTrainingManager(BaseManager):
            pass

        print(
            f"Connecting to manager at { _PROJECT_MODEL_BASE_MANAGER_ADDRESS }, authkey={_PROJECT_MODEL_AUTHKEY}..."
        )
        remote_manager = CandidateTrainingManager(
            address=_PROJECT_MODEL_BASE_MANAGER_ADDRESS,
            authkey=_PROJECT_MODEL_AUTHKEY,
            ctx=remote_context,
        )
        print(f"{_PROJECT_TESTING_PARTIAL_TEST} Created manager class successfully...")
        print(
            f"Created manager instance: type: {type(remote_manager)}, value: {remote_manager}"
        )
        print("Connecting to manager...")
        remote_manager.connect()
        print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Connected to manager successfully!")

        # Test queue access
        print("Accessing queues...")
        print("Accessing tasks queue...")
        tasks_queue = remote_manager.get_tasks_queue()
        print(f"{_PROJECT_TESTING_PARTIAL_TEST} Tasks Queue access successful!")
        print("Accessing done queue...")
        done_queue = remote_manager.get_done_queue()
        print(f"{_PROJECT_TESTING_PARTIAL_TEST} Done Queue access successful!")

        # Try to get a task
        print("Trying to get a task from the task queue...")
        try:
            task = tasks_queue.get(timeout=5)
            print(f"{_PROJECT_TESTING_PARTIAL_TEST} Retrieved task: {task[0]}")

            # Process task (simplified)
            print("Performing Simplified Processing of retrieved task...")
            result = (task[0], task[1][4], 0.5, None)  # Mock result
            print(f"{_PROJECT_TESTING_PARTIAL_TEST} Processed task {task[0]}...")
            print(f"Result: {result}")
            print("Putting result in done queue...")
            done_queue.put(result)
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Result put in done queue successfully!"
            )
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Processed task {task[0]} and returned result in Done queue"
            )

        # except mp.queues.Empty:
        except Exception as e:
            print(f"{_PROJECT_TESTING_UNSTABLE_TEST} No tasks available (timeout)")
            print(f"{_PROJECT_TESTING_UNSTABLE_TEST} Queue get error:\n{e}")

        return True

    except Exception as e:
        print(f"\n{_PROJECT_TESTING_FAILED_TEST} Manager connection test failed: {e}")
        import traceback

        print("printing traceback:")
        traceback.print_exc()
        return False


############################################################################################################################################################
# Summarize results
def summarize_results(
    test_local=False, local_success=False, test_remote=False, connection_success=False
):
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50 + "\n")
    if not test_local and not test_remote:
        print(f"\n{_PROJECT_TESTING_SKIPPED_TEST} Overall Test Result: SKIPPED")
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Local multiprocessing test was SKIPPED"
        )
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Remote multiprocessing test was SKIPPED"
        )
    elif not local_success and not connection_success:
        print(f"\n{_PROJECT_TESTING_FAILED_TEST} Overall Test Result: FAILURE")
        if not test_local:
            print(
                f"\n{_PROJECT_TESTING_SKIPPED_TEST} Local multiprocessing test was SKIPPED"
            )
        else:
            print(f"\n{_PROJECT_TESTING_FAILED_TEST} Local multiprocessing test FAILED")
        if not test_remote:
            print(
                f"\n{_PROJECT_TESTING_SKIPPED_TEST} Remote multiprocessing test was SKIPPED"
            )
        else:
            print(
                f"\n{_PROJECT_TESTING_FAILED_TEST} Remote multiprocessing test FAILED"
            )
    elif test_remote and connection_success and not test_local:
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Overall Test Result: SUCCESS")
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Local multiprocessing test was SKIPPED"
        )
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Remote multiprocessing test PASSED")
    elif test_local and local_success and not test_remote:
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Overall Test Result: SUCCESS")
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Local multiprocessing test PASSED")
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Remote multiprocessing test was SKIPPED"
        )
    elif (test_local and local_success) and connection_success:
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Overall Test Result: SUCCESS")
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Local multiprocessing test PASSED")
        print(f"\n{_PROJECT_TESTING_PASSED_TEST} Remote multiprocessing test PASSED")
    else:
        print(f"\n{_PROJECT_TESTING_UNKNOWN_TEST} Overall Test Result: UNKNOWN")
        print(f"\n{_PROJECT_TESTING_UNKNOWN_TEST} Local multiprocessing test UNKNOWN")
        print(f"\n{_PROJECT_TESTING_UNKNOWN_TEST} Remote multiprocessing test UNKNOWN")

    # Display results for local multiprocessing tests
    print("\nDisplaying results for local multiprocessing tests")
    if test_local and local_success:
        print(
            f"\n{_PROJECT_TESTING_PASSED_TEST} Local multiprocessing is working correctly!"
        )
    elif test_local:
        print(
            f"\n{_PROJECT_TESTING_FAILED_TEST} Issues found with local multiprocessing implementation"
        )
    else:
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Local multiprocessing functionality was not tested."
        )

    # Display results for remote multiprocessing tests
    print("\nDisplaying results for remote multiprocessing tests")
    if test_remote and connection_success:
        print(
            f"\n{_PROJECT_TESTING_PASSED_TEST} Remote multiprocessing is working correctly!"
        )
    elif test_remote:
        print(
            f"\n{_PROJECT_TESTING_FAILED_TEST} Issues found with remote multiprocessing implementation"
        )
    else:
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Remote multiprocessing functionality was not tested."
        )
    print("\nTesting complete.")


############################################################################################################################################################
# Main execution
############################################################################################################################################################
if __name__ == "__main__":
    print("Comprehensive Multiprocessing Test")
    print("=" * 50)

    ########################################################################################################################################################
    # Comment out the interactive prompt for automated testing
    ########################################################################################################################################################
    print(
        f"\n{_PROJECT_TESTING_UNKNOWN_TEST} Would you like to test local Multiprocessing? (y/n): ",
        end="",
    )
    test_local = input().strip().lower() == "y"
    ########################################################################################################################################################
    # Uncomment below line when skipping the interactive prompt during automated testing, to EXCLUDE local testing
    # test_local = False
    ########################################################################################################################################################
    # Uncomment below line when skipping the interactive prompt during automated testing, to INCLUDE local testing
    # test_local = True
    ########################################################################################################################################################

    if not test_local:
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Local multiprocessing functionality will not be tested."
        )
        local_success = False
    else:
        # Test 1: Local multiprocessing
        print("Testing local multiprocessing functionality...")
        local_success = test_local_multiprocessing()

        if local_success:
            print(
                f"\n{_PROJECT_TESTING_PARTIAL_TEST} Local multiprocessing test PASSED"
            )
        else:
            print(f"\n{_PROJECT_TESTING_FAILED_TEST} Local multiprocessing test FAILED")

    ########################################################################################################################################################
    # Ask user if they want to test remote functionality
    print("\nRemote functionality requires running a separate server process.")

    ########################################################################################################################################################
    # Comment out the interactive prompt for automated testing
    ########################################################################################################################################################
    print(
        f"\n{_PROJECT_TESTING_UNKNOWN_TEST} Would you like to start a manager server? (y/n): ",
        end="",
    )
    test_remote = input().strip().lower() == "y"
    ########################################################################################################################################################
    # Uncomment below line when skipping the interactive prompt during automated testing, to EXCLUDE remote testing
    # test_remote = False
    ########################################################################################################################################################
    # Uncomment below line when skipping the interactive prompt during automated testing, to INCLUDE remote testing
    # test_remote = True
    ########################################################################################################################################################

    if not test_remote:
        print(
            f"\n{_PROJECT_TESTING_SKIPPED_TEST} Remote functionality will not be tested."
        )
        connection_success = False
    else:
        # Test 2: Get manager context from network object
        print("Test 2: instantiate network object and manager context...")
        network, manager_context = initialize_network_and_context(
            candidate_pool_size=_CANDIDATE_POOL_SIZE,
            candidate_epochs=_CANDIDATE_EPOCHS,
        )  # initializes global variable for network and manager_context
        print(
            f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully initialized network and manager context for remote testing..."
        )
        print(f"Network object: type: {type(network)}, value: {network}")
        print(
            f"Manager context: type: {type(manager_context)}, value: {manager_context}"
        )

        # Test 2: Initialize manager and get test data
        print("Test 2: initializing manager and getting test data...")
        manager_tuple = initialize_manager()
        print(
            f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully initialized manager and retrieved test data for remote testing..."
        )
        (
            manager,
            x,
            y,
            residual_error,
            candidate_pool_size,
            candidate_epochs,
            candidate_training_queue_address,
        ) = manager_tuple
        print(f"Manager: type: {type(manager)}, value: {manager}")
        print(
            f"Candidate training queue address: type: {type(network.candidate_training_queue_address)}, value: {network.candidate_training_queue_address}"
        )
        print(
            f"Candidate pool size: type: {type(candidate_pool_size)}, value: {candidate_pool_size}"
        )
        print(
            f"Candidate epochs: type: {type(candidate_epochs)}, value: {candidate_epochs}"
        )
        print(f"Residual error: type: {type(residual_error)}, value: {residual_error}")
        print(f"Test data (x): type: {type(x)}, value: {x}")
        print(f"Test data (y): type: {type(y)}, value: {y}")

        # Test 2: Prepare arguments for manager server process
        print("Preparing arguments for manager server process...")
        args = (manager_tuple,)
        print(
            f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully prepared arguments for manager server process..."
        )
        print(f"Arguments details: type: {type(args)}, value: {args}")

        try:
            # Test 2: Start manager server (run in separate process)
            print("Instantiating multiprocessing manager server for remote testing...")
            # can't start a new process without the context that doesn't exist until after the CascadeCorrelationNetwork object is created
            candidate_training_server = manager_context.Process(
                target=start_manager_server, args=args
            )
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created multiprocessing manager server for remote testing..."
            )
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Completed Startup of manager server process..."
            )
        except Exception as e:
            print(
                f"\n{_PROJECT_TESTING_FAILED_TEST} Failed to create manager server process: {e}"
            )
            import traceback

            print("printing traceback:")
            traceback.print_exc()
            candidate_training_server = None

        if candidate_training_server is not None:
            print(f"{_PROJECT_TESTING_PARTIAL_TEST} Created manager server process...")
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Manager server process details: type: {type(candidate_training_server)}, value: {candidate_training_server}"
            )

            # print("Starting manager server process...")
            # candidate_training_server.start()
            # print(f"{_PROJECT_TESTING_PARTIAL_TEST} Started manager server process...")
            # manager.get_tasks_queue()  # Initialize queues before starting server
            # time.sleep(10)  # Give server time to start
            # print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Completed 10 sec wait for server to initialize...")
            # print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Successfully started multiprocessing manager server for remote testing...")
            # (success, network, manager_context) = result  # unpack the returned tuple
            # (success, manager_context) = result  # unpack the returned tuple
            # print(f"{_PROJECT_TESTING_PARTIAL_TEST} Manager server process returned success: {success}")
            # print(f"{_PROJECT_TESTING_PARTIAL_TEST} Manager server process returned network: {network}")
            # print(f"{_PROJECT_TESTING_PARTIAL_TEST} Manager server process returned manager_context: {manager_context}")

            print("Giving server time to start...")
            time.sleep(10)  # give server time to start
            print(f"{_PROJECT_TESTING_PARTIAL_TEST} Server started Successfully...")

            # Add some test tasks
            print("Adding test tasks to manager queue...")
            tasks_queue = manager.get_tasks_queue()
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully accessed tasks queue..."
            )
            print(
                f"Tasks queue details: type: {type(tasks_queue)}, value: {tasks_queue}"
            )
            print("Accessing done queue...")
            done_queue = manager.get_done_queue()
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully accessed done queue..."
            )
            print(f"Done queue details: type: {type(done_queue)}, value: {done_queue}")

            # Create candidate data for testing
            print("Creating candidate data for testing...")
            candidate_uuids = [f"test-uuid-{i}" for i in range(candidate_pool_size)]
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created candidate UUIDs..."
            )
            print(f"Candidate UUIDs: {candidate_uuids}")
            candidate_seeds = [42 + i for i in range(candidate_pool_size)]
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created candidate seeds..."
            )
            print(f"Candidate seeds: {candidate_seeds}")
            candidate_data = [
                (i, 2, "tanh", 0.1, candidate_uuids[i], candidate_seeds[i], 1000, 10)
                for i in range(candidate_pool_size)
            ]
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created candidate data..."
            )
            print(
                f"Candidate data: length: {len(candidate_data)}, sample: {candidate_data[0]}"
            )

            print("Generating training inputs and tasks...")
            training_inputs = (x, 5, y, residual_error, 0.01, 20)
            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created training inputs..."
            )
            print(
                f"Training inputs details: type: {type(training_inputs)}, value: {training_inputs}"
            )
            tasks = [
                (i, candidate_data[i], training_inputs)
                for i in range(len(candidate_pool_size))
            ]
            print(f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully created tasks...")
            print(f"Tasks details: length: {len(tasks)}, sample: {tasks[0]}")

            # Add tasks to queue
            print("Adding tasks to queue...")
            for task in tasks:
                print(f"Adding task {task[0]} to queue: {task}")
                tasks_queue.put(task)
                print(
                    f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully added task {task[0]} to queue..."
                )
                print(f"Tasks queue size: {tasks_queue.qsize()}")

            print(
                f"{_PROJECT_TESTING_PARTIAL_TEST} Successfully added tasks to queue..."
            )
            print(f"Tasks queue size: {tasks_queue.qsize()}")
            print(
                f"\n{_PROJECT_TESTING_PARTIAL_TEST} Manager server running at {candidate_training_queue_address}"
            )
            print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Added test tasks to queue")

        # # Keep server running
        # print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Manager server running...")
        # time.sleep(120)
        # print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Manager server run complete...")

        try:  # Test 2: Cleanup Testing connection to manager server
            print("Testing connection to manager server...")
            if connection_success := test_manager_connection():
                print(
                    f"\n{_PROJECT_TESTING_PASSED_TEST} Manager connection test PASSED"
                )
            else:
                print(
                    f"\n{_PROJECT_TESTING_FAILED_TEST} Manager connection test FAILED"
                )

            # Shutdown
            print("Shutting down manager server...")
            candidate_training_server.shutdown()
            print(f"\n{_PROJECT_TESTING_PARTIAL_TEST} Manager server stopped")
            print("Joining manager server process...")
            print(f"\n{_PROJECT_TESTING_PASSED_TEST} Manager server test PASSED")

        except Exception as e:
            print(f"\n{_PROJECT_TESTING_FAILED_TEST} Manager server test failed: {e}")
            import traceback

            print("printing traceback:")
            traceback.print_exc()

    # Summarize results
    summarize_results(
        test_local=test_local,
        local_success=local_success,
        test_remote=test_remote,
        connection_success=connection_success,
    )
