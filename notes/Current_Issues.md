# Current Issue Tracker for the Cascade Correlation Network prototype

```bash
+[candidate_unit.py: 533] (2025-10-15 17:37:33) [ERROR] CandidateUnit: train: Failed to display training progress: CandidateUnit._init_display_progress() takes from 1 to 2 positional arguments but 4 were given
+[candidate_unit.py: 535] (2025-10-15 17:37:33) [ERROR] CandidateUnit: train: Traceback: Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/candidate_unit/candidate_unit.py", line 531, in train
    self._init_display_progress(epoch, candidate_parameters_update, residual_error)
TypeError: CandidateUnit._init_display_progress() takes from 1 to 2 positional arguments but 4 were given


[cascade_correlation.py:1299] (25-10-15 17:37:33) [WARNING] CascadeCorrelationNetwork: _stop_workers: Worker CandidateWorker-14 did not stop, terminating
[cascade_correlation.py:1238] (25-10-15 17:37:33) [DEBUG] CascadeCorrelationNetwork: _execute_parallel_training: Stopped all workers
[cascade_correlation.py:1711] (25-10-15 17:37:33) [DEBUG] CascadeCorrelationNetwork: _stop_manager: Stopping multiprocessing manager
[cascade_correlation.py:1716] (25-10-15 17:37:33) [INFO] CascadeCorrelationNetwork: _stop_manager: Manager shutdown completed
[cascade_correlation.py:1151] (25-10-15 17:37:33) [WARNING] CascadeCorrelationNetwork: _execute_candidate_training: Parallel processing returned no results, falling back to sequential
[cascade_correlation.py:1161] (25-10-15 17:37:33) [ERROR] CascadeCorrelationNetwork: _execute_candidate_training: Error in candidate node training: Parallel processing failed to return results
[cascade_correlation.py:1163] (25-10-15 17:37:33) [ERROR] CascadeCorrelationNetwork: _execute_candidate_training: Traceback: Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1152, in _execute_candidate_training
    raise RuntimeError("Parallel processing failed to return results")
RuntimeError: Parallel processing failed to return results

[cascade_correlation.py:1165] (25-10-15 17:37:33) [WARNING] CascadeCorrelationNetwork: _execute_candidate_training: Creating dummy results for failed training
[cascade_correlation.py:1168] (25-10-15 17:37:33) [DEBUG] CascadeCorrelationNetwork: _execute_candidate_training: Obtained 50 results
[cascade_correlation.py:1017] (25-10-15 17:37:33) [DEBUG] CascadeCorrelationNetwork: train_candidates: Candidate training results: length: 50, value: [(0, None, 0.0, None), (1, None, 0.0, None), (2, None, 0.0, None), (3, None, 0.0, None), (4, None, 0.0, None), (5, None, 0.0, None), (6, None, 0.0, None), (7, None, 0.0, None), (8, None, 0.0, None), (9, None, 0.0, None), (10, None, 0.0, None), (11, None, 0.0, None), (12, None, 0.0, None), (13, None, 0.0, None), (14, None, 0.0, None), (15, None, 0.0, None), (16, None, 0.0, None), (17, None, 0.0, None), (18, None, 0.0, None), (19, None, 0.0, None), (20, None, 0.0, None), (21, None, 0.0, None), (22, None, 0.0, None), (23, None, 0.0, None), (24, None, 0.0, None), (25, None, 0.0, None), (26, None, 0.0, None), (27, None, 0.0, None), (28, None, 0.0, None), (29, None, 0.0, None), (30, None, 0.0, None), (31, None, 0.0, None), (32, None, 0.0, None), (33, None, 0.0, None), (34, None, 0.0, None), (35, None, 0.0, None), (36, None, 0.0, None), (37, None, 0.0, None), (38, None, 0.0, None), (39, None, 0.0, None), (40, None, 0.0, None), (41, None, 0.0, None), (42, None, 0.0, None), (43, None, 0.0, None), (44, None, 0.0, None), (45, None, 0.0, None), (46, None, 0.0, None), (47, None, 0.0, None), (48, None, 0.0, None), (49, None, 0.0, None)]
[cascade_correlation.py:1316] (25-10-15 17:37:33) [INFO] CascadeCorrelationNetwork: _process_training_results: Training duration: 0:03:33.076820
[cascade_correlation.py:2082] (25-10-15 17:37:33) [ERROR] CascadeCorrelationNetwork: grow_network: Caught Exception while training candidates at epoch 1/0:
Exception:
'tuple' object has no attribute 'correlation'
Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 2079, in _get_training_results
    training_results = self.train_candidates(x=x_train, y=y_train, residual_error=residual_error)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1028, in train_candidates
    training_stats = self._process_training_results(results, tasks, start_time)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1328, in _process_training_results
    results.sort(key=lambda r: (r.correlation is not None, np.abs(r.correlation)), reverse=True)
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1328, in <lambda>
    results.sort(key=lambda r: (r.correlation is not None, np.abs(r.correlation)), reverse=True)
                                ^^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'correlation'
Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 2079, in _get_training_results
    training_results = self.train_candidates(x=x_train, y=y_train, residual_error=residual_error)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1028, in train_candidates
    training_stats = self._process_training_results(results, tasks, start_time)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1328, in _process_training_results
    results.sort(key=lambda r: (r.correlation is not None, np.abs(r.correlation)), reverse=True)
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 1328, in <lambda>
    results.sort(key=lambda r: (r.correlation is not None, np.abs(r.correlation)), reverse=True)
                                ^^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'correlation'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascor.py", line 297, in <module>
    main()
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascor.py", line 277, in main
    sp.evaluate(
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/spiral_problem/spiral_problem.py", line 1281, in evaluate
    self.solve_n_spiral_problem(
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/spiral_problem/spiral_problem.py", line 1190, in solve_n_spiral_problem
    self.history = self.network.fit(self.x, self.y, max_epochs=_SPIRAL_PROBLEM_OUTPUT_EPOCHS,)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 793, in fit
    self.grow_network(
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 2009, in grow_network
    training_results = self._get_training_results(x_train=x_train, y_train=y_train, residual_error=residual_error)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/cascade_correlation/cascade_correlation.py", line 2085, in _get_training_results
    raise TrainingError from e
cascade_correlation_exceptions.cascade_correlation_exceptions.TrainingError
```
