# Debugging multiprocessing BaseManager with forkserver context

I see. The ./tests script sources a config file and runs pytest with all integration and slow tests. This takes a very long time. Let me run the tests in a way that produces output continuously:

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && cd src/tests && python -m pytest integration/test_spiral_problem.py -v --no-cov --slow --integration -x 2>&1 &
pid=$!
sleep 5
while kill -0 $pid 2>/dev/null; do
    sleep 5
done
wait $pid 2>/dev/null
echo "Tests completed"

```log
Warning: long output, 2.4k lines truncated

--- Truncated 2436 lines above this point ---
        [0.4594, 0.4284],
        [0.4516, 0.4414],
        [0.4484, 0.4438],
        [0.4479, 0.4450],
        [0.4259, 0.4786],
        [0.3891, 0.5363],
        [0.3721, 0.5619],
        [0.3699, 0.5659],
        [0.3889, 0.5402],
        [0.3467, 0.6063],
        [0.3603, 0.5874],
        [0.3632, 0.5844],
        [0.3980, 0.5363],
        [0.4130, 0.5180],
        [0.4181, 0.5080],
        [0.4598, 0.4499],
        [0.4801, 0.4166],
        [0.5280, 0.3447],
        [0.5614, 0.2902],
        [0.5852, 0.2533],
        [0.5952, 0.2350],
        [0.5863, 0.2455],
        [0.5868, 0.2392],
        [0.5705, 0.2597],
        [0.5314, 0.3158],
        [0.5229, 0.3259],
        [0.4763, 0.3927],
        [0.4467, 0.4341],
        [0.4042, 0.4964],
        [0.3557, 0.5737],
        [0.3169, 0.6331],
        [0.2837, 0.6864],
        [0.2660, 0.7175],
        [0.2603, 0.7285],
        [0.2299, 0.7793],
        [0.2686, 0.7297],
        [0.2856, 0.7085],
        [0.3213, 0.6582],
        [0.3494, 0.6197],
        [0.4149, 0.5252],
        [0.4595, 0.4593]])
[cascade_correlation.py:1154] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: train_output_layer: Final output layer training loss: 0.243668
[cascade_correlation.py:3675] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: get_uuid: Current UUID: 5f48eaf7-3d67-4f5b-94fd-9d7778b75e87
[cascade_correlation.py:3683] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: get_uuid: Returning UUID: 5f48eaf7-3d67-4f5b-94fd-9d7778b75e87
[__init__.py:47] (26-01-16 01:38:55) [DEBUG] Creating converter from 7 to 5
[__init__.py:47] (26-01-16 01:38:55) [DEBUG] Creating converter from 5 to 7
[__init__.py:47] (26-01-16 01:38:55) [DEBUG] Creating converter from 7 to 5
[__init__.py:47] (26-01-16 01:38:55) [DEBUG] Creating converter from 5 to 7
[snapshot_serializer.py:86] (26-01-16 01:38:55) [INFO] CascadeHDF5Serializer: Saving network to /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascor_snapshots/cascor_snapshot_20260116_013855_5f48eaf7-3d67-4f5b-94fd-9d7778b75e87.h5
[snapshot_serializer.py:241] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved root attributes
[cascade_correlation.py:3675] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: get_uuid: Current UUID: 5f48eaf7-3d67-4f5b-94fd-9d7778b75e87
[cascade_correlation.py:3683] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: get_uuid: Returning UUID: 5f48eaf7-3d67-4f5b-94fd-9d7778b75e87
[snapshot_serializer.py:267] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved metadata with training counters
[dataset.py:180] (26-01-16 01:38:55) [DEBUG] Creating converter from 5 to 3
[snapshot_serializer.py:304] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved configuration
[snapshot_serializer.py:332] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved architecture
[snapshot_serializer.py:368] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved parameter checksums
[snapshot_serializer.py:390] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved optimizer state
[snapshot_serializer.py:378] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved parameters
[snapshot_serializer.py:502] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved all random states (Python, NumPy, PyTorch, CUDA)
[snapshot_serializer.py:507] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved random state
[snapshot_serializer.py:519] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Saved multiprocessing state
[snapshot_serializer.py:107] (26-01-16 01:38:55) [INFO] CascadeHDF5Serializer: Successfully saved network to /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascor_snapshots/cascor_snapshot_20260116_013855_5f48eaf7-3d67-4f5b-94fd-9d7778b75e87.h5
[cascade_correlation.py:2858] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: save_to_hdf5: Successfully saved to /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascor_snapshots/cascor_snapshot_20260116_013855_5f48eaf7-3d67-4f5b-94fd-9d7778b75e87.h5
[cascade_correlation.py:2861] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: save_to_hdf5: Verifying saved HDF5 file
[snapshot_serializer.py:1146] (26-01-16 01:38:55) [DEBUG] CascadeHDF5Serializer: Format validation passed
[cascade_correlation.py:2866] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: save_to_hdf5: Verified saved HDF5 file is valid: /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascor_snapshots/cascor_snapshot_20260116_013855_5f48eaf7-3d67-4f5b-94fd-9d7778b75e87.h5
[cascade_correlation.py:2658] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: create_snapshot: Created snapshot at /home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascor_snapshots/cascor_snapshot_20260116_013855_5f48eaf7-3d67-4f5b-94fd-9d7778b75e87.h5
[cascade_correlation.py:1156] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: train_output_layer: Created network snapshot at: True
[cascade_correlation.py:969] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: fit: Initial - Train Loss: 0.243668
[cascade_correlation.py:3270] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Validated input shape: torch.Size([100, 2]), Target shape: torch.Size([100, 2])
[cascade_correlation.py:3273] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Calculating accuracy for input shape: torch.Size([100, 2]), target shape: torch.Size([100, 2])
[cascade_correlation.py:1034] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: forward: Input shape: torch.Size([100, 2])
[cascade_correlation.py:1056] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: forward: Output shape: torch.Size([100, 2])
[cascade_correlation.py:3276] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Output shape: torch.Size([100, 2]), Output: tensor([[0.4161, 0.4962],
        [0.4308, 0.4744],
        [0.4274, 0.4804],
        [0.4297, 0.4809],
        [0.4322, 0.4742],
        [0.4031, 0.5209],
        [0.4139, 0.5044],
        [0.4156, 0.5049],
        [0.4218, 0.4957],
        [0.4241, 0.4904],
        [0.4352, 0.4772],
        [0.4638, 0.4363],
        [0.4968, 0.3861],
        [0.4963, 0.3864],
        [0.4712, 0.4234],
        [0.5120, 0.3591],
        [0.5414, 0.3122],
        [0.5403, 0.3140],
        [0.5286, 0.3295],
        [0.5232, 0.3339],
        [0.5061, 0.3586],
        [0.5006, 0.3624],
        [0.4712, 0.4050],
        [0.4611, 0.4177],
        [0.4368, 0.4541],
        [0.3885, 0.5267],
        [0.3875, 0.5302],
        [0.3516, 0.5850],
        [0.3079, 0.6552],
        [0.3040, 0.6639],
        [0.3066, 0.6641],
        [0.2925, 0.6878],
        [0.3123, 0.6649],
        [0.3422, 0.6232],
        [0.3631, 0.5938],
        [0.4062, 0.5310],
        [0.4531, 0.4633],
        [0.4843, 0.4148],
        [0.5223, 0.3594],
        [0.5813, 0.2689],
        [0.5948, 0.2454],
        [0.6352, 0.1809],
        [0.6328, 0.1785],
        [0.6438, 0.1558],
        [0.6304, 0.1713],
        [0.6106, 0.1933],
        [0.5906, 0.2223],
        [0.5330, 0.3015],
        [0.4952, 0.3586],
        [0.4385, 0.4399],
        [0.4473, 0.4538],
        [0.4410, 0.4613],
        [0.4649, 0.4266],
        [0.4741, 0.4127],
        [0.4644, 0.4279],
        [0.4808, 0.4021],
        [0.4678, 0.4204],
        [0.4755, 0.4068],
        [0.4589, 0.4320],
        [0.4594, 0.4284],
        [0.4516, 0.4414],
        [0.4484, 0.4438],
        [0.4479, 0.4450],
        [0.4259, 0.4786],
        [0.3891, 0.5363],
        [0.3721, 0.5619],
        [0.3699, 0.5659],
        [0.3889, 0.5402],
        [0.3467, 0.6063],
        [0.3603, 0.5874],
        [0.3632, 0.5844],
        [0.3980, 0.5363],
        [0.4130, 0.5180],
        [0.4181, 0.5080],
        [0.4598, 0.4499],
        [0.4801, 0.4166],
        [0.5280, 0.3447],
        [0.5614, 0.2902],
        [0.5852, 0.2533],
        [0.5952, 0.2350],
        [0.5863, 0.2455],
        [0.5868, 0.2392],
        [0.5705, 0.2597],
        [0.5314, 0.3158],
        [0.5229, 0.3259],
        [0.4763, 0.3927],
        [0.4467, 0.4341],
        [0.4042, 0.4964],
        [0.3557, 0.5737],
        [0.3169, 0.6331],
        [0.2837, 0.6864],
        [0.2660, 0.7175],
        [0.2603, 0.7285],
        [0.2299, 0.7793],
        [0.2686, 0.7297],
        [0.2856, 0.7085],
        [0.3213, 0.6582],
        [0.3494, 0.6197],
        [0.4149, 0.5252],
        [0.4595, 0.4593]])
[cascade_correlation.py:3289] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Validated Output shape: torch.Size([100, 2]), Target shape: torch.Size([100, 2])
[cascade_correlation.py:3325] (26-01-16 01:38:55) [DEBUG] CascadeCorrelationNetwork: _accuracy: Input shape: torch.Size([100, 2]), Output shape: torch.Size([100, 2])
[cascade_correlation.py:3342] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: _accuracy: Calculated accuracy: 0.5000, Percentage: 50.0000%
[cascade_correlation.py:3291] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: calculate_accuracy: Calculated accuracy: 0.5000, Percentage: 50.00%
[cascade_correlation.py:979] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: fit: Initial - Train Accuracy: 0.5000
[cascade_correlation.py:985] (26-01-16 01:38:55) [INFO] CascadeCorrelationNetwork: fit: Starting main training loop with max epochs: 10, early stopping: False
+[candidate_unit.py: 158] (2026-01-16 01:38:55) [INFO] CandidateUnit: __init__: Initializing Candidate Unit with Logger class.
+[candidate_unit.py: 162] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: __init__: Candidate index: 0
+[candidate_unit.py: 166] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: __init__: Random seed: 42
+[candidate_unit.py: 168] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: __init__: Random max value: 4294967295
+[candidate_unit.py: 170] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: __init__: Random sequence max value: 10
+[candidate_unit.py: 271] (2026-01-16 01:38:55) [TRACE] CandidateUnit: _initialize_randomness: Initializing randomness for the candidate unit
+[candidate_unit.py: 273] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: _initialize_randomness: Random seed set to: 42
+[candidate_unit.py: 277] (2026-01-16 01:38:55) [VERBOSE] CandidateUnit: _initialize_randomness: Random max value set to: 10
+[candidate_unit.py: 309] (2026-01-16 01:38:55) [TRACE] CandidateUnit: _seed_random_generator: Seeding random module with seed and max value.
+[candidate_unit.py: 314] (2026-01-16 01:38:55) [TRACE] CandidateUnit: _seed_random_generator: Random seed set for random module.
+[candidate_unit.py: 321] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _seed_random_generator: Random sequence number rolled to: 6
+[candidate_unit.py: 339] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _roll_sequence_number: Rolling sequence number.
+[candidate_unit.py: 342] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _roll_sequence_number: Discarded 6 random values to roll to the desired sequence.
+[candidate_unit.py: 343] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _roll_sequence_number: Random Generator rolled for sequence number: 6
+[candidate_unit.py: 344] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _roll_sequence_number: Completed rolling of sequence number.
+[candidate_unit.py: 323] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Completed initialization of random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 279] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _initialize_randomness: Completed initialization of numpy random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 309] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Seeding random module with seed and max value.
+[candidate_unit.py: 314] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Random seed set for random module.
+[candidate_unit.py: 321] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _seed_random_generator: Random sequence number rolled to: 10
+[candidate_unit.py: 339] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _roll_sequence_number: Rolling sequence number.
+[candidate_unit.py: 342] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _roll_sequence_number: Discarded 10 random values to roll to the desired sequence.
+[candidate_unit.py: 343] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _roll_sequence_number: Random Generator rolled for sequence number: 10
+[candidate_unit.py: 344] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _roll_sequence_number: Completed rolling of sequence number.
+[candidate_unit.py: 323] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Completed initialization of random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 281] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _initialize_randomness: Completed initialization of random random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 309] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Seeding random module with seed and max value.
+[candidate_unit.py: 314] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _seed_random_generator: Random seed set for random module.
+[candidate_unit.py: 321] (2026-01-16 01:38:56) [VERBOSE] CandidateUnit: _seed_random_generator: Random sequence number rolled to: 9
+[candidate_unit.py: 339] (2026-01-16 01:38:56) [TRACE] CandidateUnit: _roll_sequence_number: Rolling sequence number.
+[candidate_unit.py: 342] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: _roll_sequence_number: Discarded 9 random values to roll to the desired sequence.
+[candidate_unit.py: 343] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: _roll_sequence_number: Random Generator rolled for sequence number: 9
+[candidate_unit.py: 344] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _roll_sequence_number: Completed rolling of sequence number.
+[candidate_unit.py: 323] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _seed_random_generator: Completed initialization of random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 283] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _initialize_randomness: Completed initialization of torch random generator with seed and sequence for the candidate unit
+[candidate_unit.py: 309] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _seed_random_generator: Seeding random module with seed and max value.
+[candidate_unit.py: 314] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _seed_random_generator: Random seed set for random module.
+[candidate_unit.py: 316] (2026-01-16 01:38:57) [WARNING] CandidateUnit: _seed_random_generator: No generator function provided, skipping random number generation and sequence rolling.
+[candidate_unit.py: 176] (2026-01-16 01:38:57) [TRACE] CandidateUnit: __init__: Initializing CandidateUnit class attributes with input parameters.
+[candidate_unit.py: 178] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Input size: 2
+[candidate_unit.py: 180] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Output size: 2
+[candidate_unit.py: 182] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Base Activation function: Tanh()
+[candidate_unit.py: 369] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _init_activation_with_derivative: Validating activation function
+[candidate_unit.py: 371] (2026-01-16 01:38:57) [DEBUG] CandidateUnit: _init_activation_with_derivative: Using activation function: Tanh()
+[candidate_unit.py: 374] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _init_activation_with_derivative: Wrapping activation function to provide its derivative.
+[candidate_unit.py: 389] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: _init_activation_with_derivative: Returning wrapped activation function: <function CandidateUnit._init_activation_with_derivative.<locals>.wrapped_activation at 0x76ce894d2980>.
+[candidate_unit.py: 392] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: _init_activation_with_derivative: Returning wrapped activation function: Type: <class 'function'>, Value: <function CandidateUnit._init_activation_with_derivative.<locals>.wrapped_activation at 0x76ce894d2980>.
+[candidate_unit.py: 393] (2026-01-16 01:38:57) [TRACE] CandidateUnit: _init_activation_with_derivative: Completed wrapping of activation function.
+[candidate_unit.py: 186] (2026-01-16 01:38:57) [DEBUG] CandidateUnit: __init__: Cached activation function wrapper
+[candidate_unit.py: 190] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Epochs: 50
+[candidate_unit.py: 192] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Max epochs: 10000
+[candidate_unit.py: 196] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Learning rate: 0.01
+[candidate_unit.py: 198] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Random value scale: 0.1
+[candidate_unit.py: 202] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Early stopping: True
+[candidate_unit.py: 204] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Patience: 10
+[candidate_unit.py: 208] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Weights: tensor([0.1523, 0.0665])
+[candidate_unit.py: 210] (2026-01-16 01:38:57) [VERBOSE] CandidateUnit: __init__: Bias: tensor([-0.1032])
+[candidate_unit.py: 213] (2026-01-16 01:38:58) [TRACE] CandidateUnit: __init__: Initializing CandidateUnit Attributes with Constants (e.g., 0.0)
+[candidate_unit.py: 1340] (2026-01-16 01:38:58) [TRACE] CandidateUnit: set_uuid: Starting to set UUID for CandidateUnit class
+[candidate_unit.py: 1160] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _generate_uuid: Inside the CandidateUnit class Generate UUID method
+[candidate_unit.py: 1162] (2026-01-16 01:38:58) [DEBUG] CandidateUnit: _generate_uuid: UUID: f24e2512-96b9-4aea-93dc-251950d144b5
+[candidate_unit.py: 1163] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _generate_uuid: Completed the CandidateUnit class Generate UUID method
+[candidate_unit.py: 1346] (2026-01-16 01:38:58) [DEBUG] CandidateUnit: set_uuid: UUID set to: f24e2512-96b9-4aea-93dc-251950d144b5
+[candidate_unit.py: 1347] (2026-01-16 01:38:58) [TRACE] CandidateUnit: set_uuid: Completed setting UUID for CandidateUnit class
+[candidate_unit.py: 223] (2026-01-16 01:38:58) [VERBOSE] CandidateUnit: __init__: UUID: f24e2512-96b9-4aea-93dc-251950d144b5
+[candidate_unit.py: 227] (2026-01-16 01:38:58) [VERBOSE] CandidateUnit: __init__: Display frequency: 10
+[candidate_unit.py: 229] (2026-01-16 01:38:58) [VERBOSE] CandidateUnit: __init__: Status frequency: 10
+[candidate_unit.py: 1176] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _init_display_progress: Inside the CandidateUnit class Initialize Display Progress method
+[utils.py: 110] (2026-01-16 01:38:58) [DEBUG] Utils: display_progress: Display Frequency: Type: <class 'int'> Value: 10
+[candidate_unit.py: 1178] (2026-01-16 01:38:58) [DEBUG] CandidateUnit: _init_display_progress: Display frequency set to: 10, Candidate Display Progress: <function display_progress.<locals>.<lambda> at 0x76ce894d3060>
+[candidate_unit.py: 1179] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _init_display_progress: Completed the CandidateUnit class Initialize Display Progress method
+[candidate_unit.py: 233] (2026-01-16 01:38:58) [VERBOSE] CandidateUnit: __init__: Candidate display progress function initialized with display frequency: 10, _candidate_display_progress = <function display_progress.<locals>.<lambda> at 0x76ce894d3060>
+[candidate_unit.py: 1192] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _init_display_status: Inside the CandidateUnit class Initialize Display Status method
+[utils.py: 110] (2026-01-16 01:38:58) [DEBUG] Utils: display_progress: Display Frequency: Type: <class 'int'> Value: 10
+[candidate_unit.py: 1194] (2026-01-16 01:38:58) [DEBUG] CandidateUnit: _init_display_status: Candidate Display Status: <function display_progress.<locals>.<lambda> at 0x76ce894d2fb0>
+[candidate_unit.py: 1195] (2026-01-16 01:38:58) [TRACE] CandidateUnit: _init_display_status: Completed the CandidateUnit class Initialize Display Status method
+[candidate_unit.py: 235] (2026-01-16 01:38:58) [VERBOSE] CandidateUnit: __init__: Candidate display status function initialized with status frequency: 10, _candidate_display_status = <function display_progress.<locals>.<lambda> at 0x76ce894d2fb0>
+[candidate_unit.py: 237] (2026-01-16 01:38:58) [DEBUG] CandidateUnit: __init__: Completed initialization of Candidate Unit
+[candidate_unit.py: 238] (2026-01-16 01:38:58) [TRACE] CandidateUnit: __init__: Completed the __init__ method for the Candidate Unit
[cascade_correlation.py:2507] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_residual_error_safe: Starting epoch 1/0
[cascade_correlation.py:2159] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Calculating residual error for input shape: torch.Size([100, 2]), target shape: torch.Size([100, 2])
[cascade_correlation.py:2161] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: After defaulting, input shape: torch.Size([100, 2]), target shape: torch.Size([100, 2])
[cascade_correlation.py:2179] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Forward pass to calculate output for residual error computation
[cascade_correlation.py:2181] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Performing forward pass without gradient tracking
[cascade_correlation.py:1034] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: forward: Input shape: torch.Size([100, 2])
[cascade_correlation.py:1056] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: forward: Output shape: torch.Size([100, 2])
[cascade_correlation.py:2183] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Forward pass completed, output shape: torch.Size([100, 2]), Output:
tensor([[0.4161, 0.4962],
        [0.4308, 0.4744],
        [0.4274, 0.4804],
        [0.4297, 0.4809],
        [0.4322, 0.4742],
        [0.4031, 0.5209],
        [0.4139, 0.5044],
        [0.4156, 0.5049],
        [0.4218, 0.4957],
        [0.4241, 0.4904],
        [0.4352, 0.4772],
        [0.4638, 0.4363],
        [0.4968, 0.3861],
        [0.4963, 0.3864],
        [0.4712, 0.4234],
        [0.5120, 0.3591],
        [0.5414, 0.3122],
        [0.5403, 0.3140],
        [0.5286, 0.3295],
        [0.5232, 0.3339],
        [0.5061, 0.3586],
        [0.5006, 0.3624],
        [0.4712, 0.4050],
        [0.4611, 0.4177],
        [0.4368, 0.4541],
        [0.3885, 0.5267],
        [0.3875, 0.5302],
        [0.3516, 0.5850],
        [0.3079, 0.6552],
        [0.3040, 0.6639],
        [0.3066, 0.6641],
        [0.2925, 0.6878],
        [0.3123, 0.6649],
        [0.3422, 0.6232],
        [0.3631, 0.5938],
        [0.4062, 0.5310],
        [0.4531, 0.4633],
        [0.4843, 0.4148],
        [0.5223, 0.3594],
        [0.5813, 0.2689],
        [0.5948, 0.2454],
        [0.6352, 0.1809],
        [0.6328, 0.1785],
        [0.6438, 0.1558],
        [0.6304, 0.1713],
        [0.6106, 0.1933],
        [0.5906, 0.2223],
        [0.5330, 0.3015],
        [0.4952, 0.3586],
        [0.4385, 0.4399],
        [0.4473, 0.4538],
        [0.4410, 0.4613],
        [0.4649, 0.4266],
        [0.4741, 0.4127],
        [0.4644, 0.4279],
        [0.4808, 0.4021],
        [0.4678, 0.4204],
        [0.4755, 0.4068],
        [0.4589, 0.4320],
        [0.4594, 0.4284],
        [0.4516, 0.4414],
        [0.4484, 0.4438],
        [0.4479, 0.4450],
        [0.4259, 0.4786],
        [0.3891, 0.5363],
        [0.3721, 0.5619],
        [0.3699, 0.5659],
        [0.3889, 0.5402],
        [0.3467, 0.6063],
        [0.3603, 0.5874],
        [0.3632, 0.5844],
        [0.3980, 0.5363],
        [0.4130, 0.5180],
        [0.4181, 0.5080],
        [0.4598, 0.4499],
        [0.4801, 0.4166],
        [0.5280, 0.3447],
        [0.5614, 0.2902],
        [0.5852, 0.2533],
        [0.5952, 0.2350],
        [0.5863, 0.2455],
        [0.5868, 0.2392],
        [0.5705, 0.2597],
        [0.5314, 0.3158],
        [0.5229, 0.3259],
        [0.4763, 0.3927],
        [0.4467, 0.4341],
        [0.4042, 0.4964],
        [0.3557, 0.5737],
        [0.3169, 0.6331],
        [0.2837, 0.6864],
        [0.2660, 0.7175],
        [0.2603, 0.7285],
        [0.2299, 0.7793],
        [0.2686, 0.7297],
        [0.2856, 0.7085],
        [0.3213, 0.6582],
        [0.3494, 0.6197],
        [0.4149, 0.5252],
        [0.4595, 0.4593]])
[cascade_correlation.py:2185] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: torch.Size([100, 2]), Residual Error:
tensor([[ 0.5839, -0.4962],
        [ 0.5692, -0.4744],
        [ 0.5726, -0.4804],
        [ 0.5703, -0.4809],
        [ 0.5678, -0.4742],
        [ 0.5969, -0.5209],
        [ 0.5861, -0.5044],
        [ 0.5844, -0.5049],
        [ 0.5782, -0.4957],
        [ 0.5759, -0.4904],
        [ 0.5648, -0.4772],
        [ 0.5362, -0.4363],
        [ 0.5032, -0.3861],
        [ 0.5037, -0.3864],
        [ 0.5288, -0.4234],
        [ 0.4880, -0.3591],
        [ 0.4586, -0.3122],
        [ 0.4597, -0.3140],
        [ 0.4714, -0.3295],
        [ 0.4768, -0.3339],
        [ 0.4939, -0.3586],
        [ 0.4994, -0.3624],
        [ 0.5288, -0.4050],
        [ 0.5389, -0.4177],
        [ 0.5632, -0.4541],
        [ 0.6115, -0.5267],
        [ 0.6125, -0.5302],
        [ 0.6484, -0.5850],
        [ 0.6921, -0.6552],
        [ 0.6960, -0.6639],
        [ 0.6934, -0.6641],
        [ 0.7075, -0.6878],
        [ 0.6877, -0.6649],
        [ 0.6578, -0.6232],
        [ 0.6369, -0.5938],
        [ 0.5938, -0.5310],
        [ 0.5469, -0.4633],
        [ 0.5157, -0.4148],
        [ 0.4777, -0.3594],
        [ 0.4187, -0.2689],
        [ 0.4052, -0.2454],
        [ 0.3648, -0.1809],
        [ 0.3672, -0.1785],
        [ 0.3562, -0.1558],
        [ 0.3696, -0.1713],
        [ 0.3894, -0.1933],
        [ 0.4094, -0.2223],
        [ 0.4670, -0.3015],
        [ 0.5048, -0.3586],
        [ 0.5615, -0.4399],
        [-0.4473,  0.5462],
        [-0.4410,  0.5387],
        [-0.4649,  0.5734],
        [-0.4741,  0.5873],
        [-0.4644,  0.5721],
        [-0.4808,  0.5979],
        [-0.4678,  0.5796],
        [-0.4755,  0.5932],
        [-0.4589,  0.5680],
        [-0.4594,  0.5716],
        [-0.4516,  0.5586],
        [-0.4484,  0.5562],
        [-0.4479,  0.5550],
        [-0.4259,  0.5214],
        [-0.3891,  0.4637],
        [-0.3721,  0.4381],
        [-0.3699,  0.4341],
        [-0.3889,  0.4598],
        [-0.3467,  0.3937],
        [-0.3603,  0.4126],
        [-0.3632,  0.4156],
        [-0.3980,  0.4637],
        [-0.4130,  0.4820],
        [-0.4181,  0.4920],
        [-0.4598,  0.5501],
        [-0.4801,  0.5834],
        [-0.5280,  0.6553],
        [-0.5614,  0.7098],
        [-0.5852,  0.7467],
        [-0.5952,  0.7650],
        [-0.5863,  0.7545],
        [-0.5868,  0.7608],
        [-0.5705,  0.7403],
        [-0.5314,  0.6842],
        [-0.5229,  0.6741],
        [-0.4763,  0.6073],
        [-0.4467,  0.5659],
        [-0.4042,  0.5036],
        [-0.3557,  0.4263],
        [-0.3169,  0.3669],
        [-0.2837,  0.3136],
        [-0.2660,  0.2825],
        [-0.2603,  0.2715],
        [-0.2299,  0.2207],
        [-0.2686,  0.2703],
        [-0.2856,  0.2915],
        [-0.3213,  0.3418],
        [-0.3494,  0.3803],
        [-0.4149,  0.4748],
        [-0.4595,  0.5407]])
[cascade_correlation.py:2188] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: calculate_residual_error: Calculated residual error, shape: torch.Size([100, 2]), Residual Error:
tensor([[ 0.5839, -0.4962],
        [ 0.5692, -0.4744],
        [ 0.5726, -0.4804],
        [ 0.5703, -0.4809],
        [ 0.5678, -0.4742],
        [ 0.5969, -0.5209],
        [ 0.5861, -0.5044],
        [ 0.5844, -0.5049],
        [ 0.5782, -0.4957],
        [ 0.5759, -0.4904],
        [ 0.5648, -0.4772],
        [ 0.5362, -0.4363],
        [ 0.5032, -0.3861],
        [ 0.5037, -0.3864],
        [ 0.5288, -0.4234],
        [ 0.4880, -0.3591],
        [ 0.4586, -0.3122],
        [ 0.4597, -0.3140],
        [ 0.4714, -0.3295],
        [ 0.4768, -0.3339],
        [ 0.4939, -0.3586],
        [ 0.4994, -0.3624],
        [ 0.5288, -0.4050],
        [ 0.5389, -0.4177],
        [ 0.5632, -0.4541],
        [ 0.6115, -0.5267],
        [ 0.6125, -0.5302],
        [ 0.6484, -0.5850],
        [ 0.6921, -0.6552],
        [ 0.6960, -0.6639],
        [ 0.6934, -0.6641],
        [ 0.7075, -0.6878],
        [ 0.6877, -0.6649],
        [ 0.6578, -0.6232],
        [ 0.6369, -0.5938],
        [ 0.5938, -0.5310],
        [ 0.5469, -0.4633],
        [ 0.5157, -0.4148],
        [ 0.4777, -0.3594],
        [ 0.4187, -0.2689],
        [ 0.4052, -0.2454],
        [ 0.3648, -0.1809],
        [ 0.3672, -0.1785],
        [ 0.3562, -0.1558],
        [ 0.3696, -0.1713],
        [ 0.3894, -0.1933],
        [ 0.4094, -0.2223],
        [ 0.4670, -0.3015],
        [ 0.5048, -0.3586],
        [ 0.5615, -0.4399],
        [-0.4473,  0.5462],
        [-0.4410,  0.5387],
        [-0.4649,  0.5734],
        [-0.4741,  0.5873],
        [-0.4644,  0.5721],
        [-0.4808,  0.5979],
        [-0.4678,  0.5796],
        [-0.4755,  0.5932],
        [-0.4589,  0.5680],
        [-0.4594,  0.5716],
        [-0.4516,  0.5586],
        [-0.4484,  0.5562],
        [-0.4479,  0.5550],
        [-0.4259,  0.5214],
        [-0.3891,  0.4637],
        [-0.3721,  0.4381],
        [-0.3699,  0.4341],
        [-0.3889,  0.4598],
        [-0.3467,  0.3937],
        [-0.3603,  0.4126],
        [-0.3632,  0.4156],
        [-0.3980,  0.4637],
        [-0.4130,  0.4820],
        [-0.4181,  0.4920],
        [-0.4598,  0.5501],
        [-0.4801,  0.5834],
        [-0.5280,  0.6553],
        [-0.5614,  0.7098],
        [-0.5852,  0.7467],
        [-0.5952,  0.7650],
        [-0.5863,  0.7545],
        [-0.5868,  0.7608],
        [-0.5705,  0.7403],
        [-0.5314,  0.6842],
        [-0.5229,  0.6741],
        [-0.4763,  0.6073],
        [-0.4467,  0.5659],
        [-0.4042,  0.5036],
        [-0.3557,  0.4263],
        [-0.3169,  0.3669],
        [-0.2837,  0.3136],
        [-0.2660,  0.2825],
        [-0.2603,  0.2715],
        [-0.2299,  0.2207],
        [-0.2686,  0.2703],
        [-0.2856,  0.2915],
        [-0.3213,  0.3418],
        [-0.3494,  0.3803],
        [-0.4149,  0.4748],
        [-0.4595,  0.5407]])
[cascade_correlation.py:2509] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_residual_error_safe: Epoch 0, Residual Error: 0.049585
[cascade_correlation.py:2399] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: grow_network: Epoch 0, Residual Error: 0.049585
[cascade_correlation.py:1232] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _prepare_candidate_input: Candidate input shape: torch.Size([100, 2])
[cascade_correlation.py:1233] (26-01-16 01:38:59) [INFO] CascadeCorrelationNetwork: _prepare_candidate_input: Hidden units: 0
[cascade_correlation.py:1184] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: train_candidates: Prepared candidate input shape: torch.Size([100, 2])
[cascade_correlation.py:1267] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _generate_candidate_tasks: Created 16 training tasks
[cascade_correlation.py:1188] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: train_candidates: Generated 16 candidate training tasks.
[cascade_correlation.py:1276] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: CPU count: 16
[cascade_correlation.py:1277] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Candidate pool size: 16
[cascade_correlation.py:1282] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Affinity CPU count: 16
[cascade_correlation.py:1296] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Using 15 processes
[cascade_correlation.py:1192] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: train_candidates: Optimal process count for training: 15
[cascade_correlation.py:1197] (26-01-16 01:38:59) [INFO] CascadeCorrelationNetwork: train_candidates: Executing candidate training with 15 processes.
[cascade_correlation.py:1308] (26-01-16 01:38:59) [INFO] CascadeCorrelationNetwork: _execute_candidate_training: Training 16 candidates with 15 processes
[cascade_correlation.py:1310] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _execute_candidate_training: Adjusted process count to: 15
[cascade_correlation.py:1313] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _execute_candidate_training: Using 15 processes
[cascade_correlation.py:1348] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _execute_parallel_training: Using multiprocessing
[cascade_correlation.py:1276] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: CPU count: 16
[cascade_correlation.py:1277] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Candidate pool size: 16
[cascade_correlation.py:1282] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Affinity CPU count: 16
[cascade_correlation.py:1296] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _calculate_optimal_process_count: Using 15 processes
[cascade_correlation.py:1355] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _execute_parallel_training: Starting the manager server
[cascade_correlation.py:2017] (26-01-16 01:38:59) [DEBUG] CascadeCorrelationNetwork: _start_manager: Starting multiprocessing manager
[cascade_correlation.py:2034] (26-01-16 01:39:01) [ERROR] CascadeCorrelationNetwork: _start_manager: Failed to start manager:
[cascade_correlation.py:1326] (26-01-16 01:39:01) [ERROR] CascadeCorrelationNetwork: _execute_candidate_training: Error in candidate node training:
[cascade_correlation.py:1328] (26-01-16 01:39:01) [ERROR] CascadeCorrelationNetwork: _execute_candidate_training: Traceback: Traceback (most recent call last):
  File "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py", line 1314, in _execute_candidate_training
    results = self._execute_parallel_training(tasks, process_count)
  File "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py", line 1356, in _execute_parallel_training
    self._start_manager()
    ~~~~~~~~~~~~~~~~~~~^^
  File "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py", line 2027, in _start_manager
    self._manager.start()  # Non-blocking - server runs in background
    ~~~~~~~~~~~~~~~~~~~^^
  File "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py", line 255, in start
    return super().start(initializer=initializer, initargs=initargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/managers.py", line 570, in start
    self._address = reader.recv()
                    ~~~~~~~~~~~^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 256, in recv
    buf = self._recv_bytes()
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 447, in _recv_bytes
    buf = self._recv(4)
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 416, in _recv
    raise EOFError
EOFError

[cascade_correlation.py:1331] (26-01-16 01:39:01) [WARNING] CascadeCorrelationNetwork: _execute_candidate_training: Creating dummy results for failed training
[cascade_correlation.py:1333] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: _execute_candidate_training: Obtained 16 results

[cascade_correlation.py:1199] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: train_candidates: Candidate training results: length: 16, value: [CandidateTrainingResult(candidate_id=0, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=1, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=2, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=Fal
se, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=3, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=4, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=5, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=6, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=7, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=8, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=9, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=10, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=11, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candida
te_id=12, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=13, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=14, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=15, candidate_uuid=None, correlation=
0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.')]

[cascade_correlation.py:1545] (26-01-16 01:39:01) [INFO] CascadeCorrelationNetwork: _process_training_results: Training duration: 0:00:02.175819
[cascade_correlation.py:1559] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: _process_training_results: Sorted 16 results

[cascade_correlation.py:1595] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: _process_training_results: Processed results: TrainingResults(epochs_completed=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], candidate_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], candidate_uuids=[], correlations=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], candidate_objects=[], best_candidate_id=0, best_candidate_uuid=None, best_correlation=0.0, best_candidate=None, success_count=0, successful_candidates=0, failed_count=16, error_messages={0: 'Candidate ID 0 (UUID: None): Invalid candidate data', 1: 'Candidate ID 1 (UUID: None): Invalid candidate data', 2: 'Candidate ID 2 (UUID: None): Invalid candidate data', 3: 'Candidate ID 3 (UUID: None): Invalid candidate data', 4: 'Candidate ID 4 (UUID: None): Invalid candidate data', 5: 'Candidate ID 5 (UUID: None): Invalid candidate data', 6: 'Candidate ID 6 (UUID: None): Invalid candidate data', 7: 'Candidate ID 7 (UUID: None): Invalid candidate data', 8: 'Candidate ID 8 (UUID: None): Invalid candidate data', 9: 'Candidate ID 9 (UUID: None): Invalid candidate data', 10: 'Candidate ID 10 (UUID: None): Invalid candidate data', 11: 'Candidate ID 11 (UUID: None): Invalid candidate data', 12: 'Candidate ID 12 (UUID: None): Invalid candidate data', 13: 'Candidate ID 13 (UUID: None): Invalid candidate data', 14: 'Candidate ID 14 (UUID: None): Invalid candidate data', 15: 'Candidate ID 15 (UUID: None): Invalid candidate data'}, max_correlation=0.0, start_time=datetime.datetime(2026, 1, 16, 1, 38, 59, 75533), end_time=datetime.datetime(2026, 1, 16, 1, 39, 1, 251352))

[cascade_correlation.py:2546] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: _get_training_results: Training Results - Epoch 0, Successful: 0, Failed: 16
[cascade_correlation.py:2403] (26-01-16 01:39:01) [WARNING] CascadeCorrelationNetwork: grow_network: Training results are None or best candidate is None, stopping growth of the network.
[cascade_correlation.py:2464] (26-01-16 01:39:01) [WARNING] CascadeCorrelationNetwork: grow_network: Validation failed at epoch 1/10.
[cascade_correlation.py:2473] (26-01-16 01:39:01) [INFO] CascadeCorrelationNetwork: grow_network: Finished training after 1 epochs. Total hidden units: 0
[cascade_correlation.py:2474] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: grow_network: Final history:
{'train_loss': [0.2436680942773819], 'value_loss': [], 'train_accuracy': [0.5], 'value_accuracy': [], 'hidden_units_added': []}
[cascade_correlation.py:1010] (26-01-16 01:39:01) [INFO] CascadeCorrelationNetwork: fit: Training completed.
[cascade_correlation.py:1011] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: fit: Final history:
{'train_loss': [0.2436680942773819], 'value_loss': [], 'train_accuracy': [0.5], 'value_accuracy': [], 'hidden_units_added': [{'correlation': 0.0, 'weights': [], 'bias': []}]}
[cascade_correlation.py:3270] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Validated input shape: torch.Size([100, 2]), Target shape: torch.Size([100, 2])
[cascade_correlation.py:3273] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Calculating accuracy for input shape: torch.Size([100, 2]), target shape: torch.Size([100, 2])
[cascade_correlation.py:1034] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: forward: Input shape: torch.Size([100, 2])
[cascade_correlation.py:1056] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: forward: Output shape: torch.Size([100, 2])

[cascade_correlation.py:3276] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Output shape: torch.Size([100, 2]), Output: tensor([[0.4161, 0.4962],
        [0.4308, 0.4744],
        [0.4274, 0.4804],
        [0.4297, 0.4809],
        [0.4322, 0.4742],
        [0.4031, 0.5209],
        [0.4139, 0.5044],
        [0.4156, 0.5049],
        [0.4218, 0.4957],
        [0.4241, 0.4904],
        [0.4352, 0.4772],
        [0.4638, 0.4363],
        [0.4968, 0.3861],
        [0.4963, 0.3864],
        [0.4712, 0.4234],
        [0.5120, 0.3591],
        [0.5414, 0.3122],
        [0.5403, 0.3140],
        [0.5286, 0.3295],
        [0.5232, 0.3339],
        [0.5061, 0.3586],
        [0.5006, 0.3624],
        [0.4712, 0.4050],
        [0.4611, 0.4177],
        [0.4368, 0.4541],
        [0.3885, 0.5267],
        [0.3875, 0.5302],
        [0.3516, 0.5850],
        [0.3079, 0.6552],
        [0.3040, 0.6639],
        [0.3066, 0.6641],
        [0.2925, 0.6878],
        [0.3123, 0.6649],
        [0.3422, 0.6232],
        [0.3631, 0.5938],
        [0.4062, 0.5310],
        [0.4531, 0.4633],
        [0.4843, 0.4148],
        [0.5223, 0.3594],
        [0.5813, 0.2689],
        [0.5948, 0.2454],
        [0.6352, 0.1809],
        [0.6328, 0.1785],
        [0.6438, 0.1558],
        [0.6304, 0.1713],
        [0.6106, 0.1933],
        [0.5906, 0.2223],
        [0.5330, 0.3015],
        [0.4952, 0.3586],
        [0.4385, 0.4399],
        [0.4473, 0.4538],
        [0.4410, 0.4613],
        [0.4649, 0.4266],
        [0.4741, 0.4127],
        [0.4644, 0.4279],
        [0.4808, 0.4021],
        [0.4678, 0.4204],
        [0.4755, 0.4068],
        [0.4589, 0.4320],
        [0.4594, 0.4284],
        [0.4516, 0.4414],
        [0.4484, 0.4438],
        [0.4479, 0.4450],
        [0.4259, 0.4786],
        [0.3891, 0.5363],
        [0.3721, 0.5619],
        [0.3699, 0.5659],
        [0.3889, 0.5402],
        [0.3467, 0.6063],
        [0.3603, 0.5874],
        [0.3632, 0.5844],
        [0.3980, 0.5363],
        [0.4130, 0.5180],
        [0.4181, 0.5080],
        [0.4598, 0.4499],
        [0.4801, 0.4166],
        [0.5280, 0.3447],
        [0.5614, 0.2902],
        [0.5852, 0.2533],
        [0.5952, 0.2350],
        [0.5863, 0.2455],
        [0.5868, 0.2392],
        [0.5705, 0.2597],
        [0.5314, 0.3158],
        [0.5229, 0.3259],
        [0.4763, 0.3927],
        [0.4467, 0.4341],
        [0.4042, 0.4964],
        [0.3557, 0.5737],
        [0.3169, 0.6331],
        [0.2837, 0.6864],
        [0.2660, 0.7175],
        [0.2603, 0.7285],
        [0.2299, 0.7793],
        [0.2686, 0.7297],
        [0.2856, 0.7085],
        [0.3213, 0.6582],
        [0.3494, 0.6197],
        [0.4149, 0.5252],
        [0.4595, 0.4593]])
[cascade_correlation.py:3289] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: calculate_accuracy: Validated Output shape: torch.Size([100, 2]), Target shape: torch.Size([100, 2])
[cascade_correlation.py:3325] (26-01-16 01:39:01) [DEBUG] CascadeCorrelationNetwork: _accuracy: Input shape: torch.Size([100, 2]), Output shape: torch.Size([100, 2])
[cascade_correlation.py:3342] (26-01-16 01:39:01) [INFO] CascadeCorrelationNetwork: _accuracy: Calculated accuracy: 0.5000, Percentage: 50.0000%
[cascade_correlation.py:3291] (26-01-16 01:39:01) [INFO] CascadeCorrelationNetwork: calculate_accuracy: Calculated accuracy: 0.5000, Percentage: 50.00%

```

----------------------------- Captured stderr call -----------------------------
Process CandidateTrainingManager-1:
Traceback (most recent call last):
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/process.py", line 320, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/managers.py", line 595, in _run_server
    server = cls._Server(registry, address, authkey, serializer)
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/managers.py", line 160, in __init__
    self.listener = Listener(address=address, backlog=128)
                    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 481, in __init__
    self._listener = SocketListener(address, family, backlog)
                     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 625, in __init__
    self._socket.bind(address)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^
OSError: [Errno 98] Address already in use
------------------------------ Captured log call -------------------------------
DEBUG    h5py._conv:__init__.py:47 Creating converter from 7 to 5
DEBUG    h5py._conv:__init__.py:47 Creating converter from 5 to 7
DEBUG    h5py._conv:__init__.py:47 Creating converter from 7 to 5
DEBUG    h5py._conv:__init__.py:47 Creating converter from 5 to 7
DEBUG    h5py._conv:dataset.py:180 Creating converter from 5 to 3
=========================== short test summary info ============================
FAILED integration/test_spiral_problem.py::TestSpiralProblemBasic::test_2_spiral_learning
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
============================== 1 failed in 9.83s ===============================
Tests completed
`````

I can see the issue now. Even with port 0, there's still an "Address already in use" error.
Looking at the logs more closely, the issue is:

```bash
OSError: [Errno 98] Address already in use
```

But wait - the address passed is ('127.0.0.1', 0).
When port is 0, the OS should allocate a free port.
The fact that it's still failing suggests a different problem.
Let me look at the _start_manager code more closely - the BaseManager may not support port 0 properly, or there's a problem with how the address is passed.

Let me look at the multiprocessing manager implementation:

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c "

```python
from multiprocessing.managers import BaseManager
import queue

def get_queue():
    return queue.Queue()

class TestManager(BaseManager):
    pass

TestManager.register('get_queue', callable=get_queue)

# Test with dynamic port
manager = TestManager(address=('127.0.0.1', 0), authkey=b'test')
manager.start()
print(f'Manager started at: {manager.address}')
manager.shutdown()
print('Manager shutdown successfully')
"
```

Traceback (most recent call last):
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/forkserver.py", line 340, in main
    code = _serve_one(child_r, fds,
                      unused_fds,
                      old_handlers)
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/forkserver.py", line 380, in _serve_one
    code = spawn._main(child_r, parent_sentinel)
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
AttributeError: module '__main__' has no attribute 'TestManager'
Traceback (most recent call last):
  File "<string>", line 15, in <module>
    manager.start()
    ~~~~~~~~~~~~~^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/managers.py", line 570, in start
    self._address = reader.recv()
                    ~~~~~~~~~~~^^
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 256, in recv
    buf = self._recv_bytes()
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 447, in _recv_bytes
    buf = self._recv(4)
  File "/opt/miniforge3/envs/JuniperCascor/lib/python3.14/multiprocessing/connection.py", line 416, in _recv
    raise EOFError
EOFError
```

`````

Interesting - the issue is related to pickling the Manager class across processes. The forkserver context can't find TestManager because it's defined in __main__. Let me check what context is being used:

```bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c "
import multiprocessing as mp
print('Default start method:', mp.get_start_method())
"

Default start method: forkserver
```

The default start method is forkserver.
This is causing issues.
When using forkserver, Python preloads modules but the custom Manager class may not be properly registered.
Let me check what the code is trying to use as the multiprocessing context:

I see the issue.
The forkserver context has issues with multiprocessing managers in Python 3.14.
Let me look at the relevant code:

__src/cascade_correlation/ cascade_correlation.py L440-490:__

```python
[... omitted lines 1 to 439 ...]
        """Initialize activation function components."""
        self.activation_function_name = ( self.config.activation_function_name or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_NAME)
        self.activation_functions_dict = ( self.config.activation_functions_dict or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTIONS_DICT)
        self.activation_fn_no_diff = ( self.activation_functions_dict.get( self.activation_function_name, _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT,) or _CASCADE_CORRELATION_NETWORK_ACTIVATION_FUNCTION_DEFAULT)
        self.activation_fn = self._init_activation_with_derivative( self.activation_fn_no_diff)

    def _init_multiprocessing(self) -> None:
        """Initialize multiprocessing context and manager attributes."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_multiprocessing: Initializing multiprocessing components")

        # Initialize multiprocessing context
        self._mp_ctx = mp.get_context("forkserver")
        try:
            self._mp_ctx.set_forkserver_preload(
                "os",
                "uuid",
                "torch",
                "numpy",
                "random",
                "logging",
                "datetime",
                "typing.Optional",
                "utils.utils.display_progress",
                "log_config.logger.logger.Logger",
            )
        except Exception as e:
            self.logger.warning( f"CascadeCorrelationNetwork: _init_multiprocessing: Failed to set forkserver preload: {e}")

        # Initialize manager attributes
        self._manager = None
        self._task_queue = None
        self._result_queue = None

        # Initialize multiprocessing config values
        self.candidate_training_queue_authkey = ( self.config.candidate_training_queue_authkey)
        self.candidate_training_queue_address = ( self.config.candidate_training_queue_address)
        self.candidate_training_tasks_queue_timeout = ( self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT)
        self.candidate_training_shutdown_timeout = ( self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT)
        self.candidate_training_context = ( mp.get_context(self.config.candidate_training_context_type) or _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT)
        self.logger.debug( "CascadeCorrelationNetwork: _init_multiprocessing: Multiprocessing components initialized")

    def _init_display_components(self) -> None:
        """Initialize display and plotting components."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_display_components: Initializing display components")

        # Initialize display parameters
        self.display_frequency = ( self.config.display_frequency or _CASCADE_CORRELATION_NETWORK_CANDIDATE_DISPLAY_FREQUENCY)
        self.epoch_display_frequency = ( self.config.epoch_display_frequency or _CASCADE_CORRELATION_NETWORK_EPOCH_DISPLAY_FREQUENCY)
        self.generate_plots = ( self.config.generate_plots or _CASCADE_CORRELATION_NETWORK_GENERATE_PLOTS)
        self.status_display_frequency = ( self.config.status_display_frequency or _CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY)
        self.candidate_display_frequency = ( self.config.candidate_display_frequency or _CASCADE_CORRELATION_NETWORK_DISPLAY_FREQUENCY)
[... omitted lines 491 to 3768 ...]
```

__The issue is multi-faceted:__

The forkserver context is being used hardcoded in _init_multiprocessing
The BaseManager with forkserver has issues with custom managers

Let me try changing the default context to fork on Linux, which is more compatible:

__src/constants/constants_model/ constants_model.py:__

```python
# _PROJECT_MODEL_AUTHKEY = b'Juniper_Cascade_Correlation_Multiprocessing_Authkey'
# _PROJECT_MODEL_AUTHKEY = 'Juniper_Cascade_Correlation_Multiprocessing_Authkey'
_PROJECT_MODEL_AUTHKEY = "Juniper_Cascade_Correlation_Multiprocessing_Authkey"

####################################################################################################################################################################################################################
_PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT = "forkserver"    # ORIGINAL LINE      \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
# Use 'fork' context on Linux for better compatibility with BaseManager
# 'forkserver' has issues with custom Manager classes in Python 3.14
_PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT = "fork"          # NEW LINE           \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
####################################################################################################################################################################################################################

# Define base manager server address and port for remote multiprocessing shared queues
# _PROJECT_MODEL_BASE_MANAGER_ADDRESS = 'localhost'
_PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP = '127.0.0.1'
```

Now I also need to update the hardcoded forkserver in the _init_multiprocessing method:

__src/cascade_correlation/ cascade_correlation.py L446-480:__

```python
[... omitted lines 1 to 445 ...]
    def _init_multiprocessing(self) -> None:
        """Initialize multiprocessing context and manager attributes."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_multiprocessing: Initializing multiprocessing components")

        # Initialize multiprocessing context
        self._mp_ctx = mp.get_context("forkserver")
        try:
            self._mp_ctx.set_forkserver_preload(
                "os",
                "uuid",
                "torch",
                "numpy",
                "random",
                "logging",
                "datetime",
                "typing.Optional",
                "utils.utils.display_progress",
                "log_config.logger.logger.Logger",
            )
        except Exception as e:
            self.logger.warning( f"CascadeCorrelationNetwork: _init_multiprocessing: Failed to set forkserver preload: {e}")

        # Initialize manager attributes
        self._manager = None
        self._task_queue = None
        self._result_queue = None

        # Initialize multiprocessing config values
        self.candidate_training_queue_authkey = ( self.config.candidate_training_queue_authkey)
        self.candidate_training_queue_address = ( self.config.candidate_training_queue_address)
        self.candidate_training_tasks_queue_timeout = ( self.config.candidate_training_task_queue_timeout or _CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT)
        self.candidate_training_shutdown_timeout = ( self.config.candidate_training_shutdown_timeout or _CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT)
        self.candidate_training_context = ( mp.get_context(self.config.candidate_training_context_type) or _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT)
        self.logger.debug( "CascadeCorrelationNetwork: _init_multiprocessing: Multiprocessing components initialized")

[... omitted lines 481 to 3768 ...]
```



```python
    def _init_multiprocessing(self) -> None:
        """Initialize multiprocessing context and manager attributes."""
        self.logger.trace( "CascadeCorrelationNetwork: _init_multiprocessing: Initializing multiprocessing components")

####################################################################################################################################################################################################################
        # Initialize multiprocessing context
        self._mp_ctx = mp.get_context("forkserver")                           # ORIGINAL LINE      \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
        # Initialize multiprocessing context using configured context type
        # Use 'fork' context for better compatibility with BaseManager on Linux
        context_type = self.config.candidate_training_context_type or "fork"  # NEW LINE           \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
        self._mp_ctx = mp.get_context(context_type)                           # NEW LINE           \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
        
        # Only set forkserver preload if using forkserver context
        if context_type == "forkserver":                                      # NEW LINE           \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
####################################################################################################################################################################################################################

            try:
                self._mp_ctx.set_forkserver_preload(
                    "os",
                    "uuid",
```

The issue must be something else in the actual code. Let me check if candidate_training_queue_address is being properly set to ('127.0.0.1', 0):

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c

```python
import sys
sys.path.insert(0, 'src')

from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig()
print(f'Address: {config.candidate_training_queue_address}')
print(f'Context: {config.candidate_training_context_type}')
```

Address: 127.0.0.1
Context: fork
`````

The address is just the string '127.0.0.1', not a tuple ('127.0.0.1', 0). Let me check the constants:

```python
#[... omitted lines 1 to 664 ...]
_CASCADE_CORRELATION_NETWORK_RANDOM_VALUE_SCALE = _SPIRAL_PROBLEM_RANDOM_VALUE_SCALE
_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR = _HDF5_PROJECT_SNAPSHOTS_DIR

# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_AUTHKEY = _PROJECT_MODEL_AUTHKEY
_CASCADE_CORRELATION_NETWORK_AUTHKEY = _PROJECT_MODEL_AUTHKEY
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_PORT = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT
# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_TIMEOUT = _PROJECT_MODEL_BASE_MANAGER_TIMEOUT

_CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY = _PROJECT_MODEL_TARGET_ACCURACY
_CASCADE_CORRELATION_NETWORK_STATUS_DISPLAY_FREQUENCY = _SPIRAL_PROBLEM_STATUS_DISPLAY_FREQUENCY
# _CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_SLEEPYTIME = _PROJECT_MODEL_CANDIDATE_TRAINING_SLEEPYTIME
_CASCADE_CORRELATION_NETWORK_SHUTDOWN_TIMEOUT = _PROJECT_MODEL_SHUTDOWN_TIMEOUT
_CASCADE_CORRELATION_NETWORK_TASK_QUEUE_TIMEOUT = _PROJECT_MODEL_TASK_QUEUE_TIMEOUT
_CASCADE_CORRELATION_NETWORK_WORKER_STANDBY_SLEEPYTIME = _PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME
_CASCADE_CORRELATION_NETWORK_CANDIDATE_TRAINING_CONTEXT = _PROJECT_CANDIDATE_TRAINING_CONTEXT


#####################################################################################################################################################################################################
# Define constants for the CascadeCorrelationNetwork class, logging Configuration
_CASCADE_CORRELATION_NETWORK_LOG_DATE_FORMAT = _SPIRAL_PROBLEM_LOG_DATE_FORMAT
_CASCADE_CORRELATION_NETWORK_LOG_FILE_NAME = _SPIRAL_PROBLEM_LOG_FILE_NAME
_CASCADE_CORRELATION_NETWORK_LOG_FILE_PATH = _SPIRAL_PROBLEM_LOG_FILE_PATH

_CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING = _SPIRAL_PROBLEM_LOG_FORMATTER_STRING
_CASCADE_CORRELATION_NETWORK_LOG_FORMATTER_STRING_CONSOLE = _SPIRAL_PROBLEM_LOG_FORMATTER_STRING_CONSOLE
#[... omitted lines 691 to 964 ...]
```

I see the issue! On line 670:

```python
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP
```

This is setting _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS to just the IP string '127.0.0.1', not to the tuple ('127.0.0.1', 0). It should use _PROJECT_MODEL_BASE_MANAGER_ADDRESS (the tuple).

__src/constants/ constants.py__

```python
_CASCADE_CORRELATION_NETWORK_HDF5_PROJECT_SNAPSHOTS_DIR = _HDF5_PROJECT_SNAPSHOTS_DIR

# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_AUTHKEY = _PROJECT_MODEL_AUTHKEY
_CASCADE_CORRELATION_NETWORK_AUTHKEY = _PROJECT_MODEL_AUTHKEY

####################################################################################################################################################################################################################
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP  # OLD_LINE  \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
# Use the full address tuple (IP, port) - port 0 means dynamic allocation
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_ADDRESS = _PROJECT_MODEL_BASE_MANAGER_ADDRESS     # NEW_LINE  \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-
####################################################################################################################################################################################################################
_CASCADE_CORRELATION_NETWORK_BASE_MANAGER_PORT = _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT
# _CASCADE_CORRELATION_NETWORK_BASE_MANAGER_TIMEOUT = _PROJECT_MODEL_BASE_MANAGER_TIMEOUT

_CASCADE_CORRELATION_NETWORK_TARGET_ACCURACY = _PROJECT_MODEL_TARGET_ACCURACY
```

Now I need to also import _PROJECT_MODEL_BASE_MANAGER_ADDRESS in constants.py:

```python
#[... omitted lines 1 to 49 ...]
    _PROJECT_MODEL_EPOCHS_MAX,
    _PROJECT_MODEL_OUTPUT_EPOCHS,
    _PROJECT_MODEL_MAX_HIDDEN_UNITS,
    _PROJECT_MODEL_CORRELATION_THRESHOLD,
    _PROJECT_MODEL_PATIENCE,
    _PROJECT_MODEL_STATUS_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_EPOCH_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_AUTHKEY,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT,
    _PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME,
    _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
    _PROJECT_MODEL_SHUTDOWN_TIMEOUT,
    _PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT,
    _PROJECT_MODEL_TARGET_ACCURACY,
)

from constants.constants_candidates.constants_candidates import (
    _PROJECT_MODEL_CANDIDATE_EARLY_STOPPING,
    _PROJECT_MODEL_CANDIDATE_PATIENCE,
    _PROJECT_MODEL_CANDIDATE_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_CANDIDATE_POOL_SIZE,
    _PROJECT_MODEL_CANDIDATE_EPOCHS,
    _PROJECT_MODEL_CANDIDATE_UNIT_LEARNING_RATE,
)
#[... omitted lines 76 to 965 ...]
```

I need to add _PROJECT_MODEL_BASE_MANAGER_ADDRESS to the import:

__src/constants/ constants.py__

```python
    _PROJECT_MODEL_PATIENCE,
    _PROJECT_MODEL_STATUS_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_EPOCH_DISPLAY_FREQUENCY,
    _PROJECT_MODEL_AUTHKEY,

####################################################################################################################################################################################################################
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS,
####################################################################################################################################################################################################################

    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP,
    _PROJECT_MODEL_BASE_MANAGER_ADDRESS_PORT,
    _PROJECT_MODEL_WORKER_STANDBY_SLEEPYTIME,
    _PROJECT_MODEL_TASK_QUEUE_TIMEOUT,
```

Now let me verify the config is correct:

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c

```python
import sys
sys.path.insert(0, 'src')

from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig()
print(f'Address: {config.candidate_training_queue_address}')
print(f'Context: {config.candidate_training_context_type}')
print(f'Type: {type(config.candidate_training_queue_address)}')
```

Address: ('127.0.0.1', 0)
Context: fork
Type: <class 'tuple'>
`````

Excellent! Now the address is the correct tuple ('127.0.0.1', 0). Let me run the test again:

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && timeout 120 python src/main.py 2>&1 | grep -E "Manager|Error|error|Failed|started|correlation|candidate|hidden" | head -50

```log
+[logger.py: 498] (2026-01-16 01:52:32) [DEBUG] Logger: __init__: Configuring custom log levels for Logger class with names: ['FATAL', 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'VERBOSE', 'TRACE'], numbers: {'FATAL': 60, 'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10, 'VERBOSE': 5, 'TRACE': 1}, methods: {'FATAL': 'fatal', 'CRITICAL': 'critical', 'ERROR': 'error', 'WARNING': 'warning', 'INFO': 'info', 'DEBUG': 'debug', 'VERBOSE': 'verbose', 'TRACE': 'trace'}
+[logger.py: 631] (2026-01-16 01:52:32) [DEBUG] Logger: __init_custom_log_levels: Log Level Methods Dict: {'FATAL': 'fatal', 'CRITICAL': 'critical', 'ERROR': 'error', 'WARNING': 'warning', 'INFO': 'info', 'DEBUG': 'debug', 'VERBOSE': 'verbose', 'TRACE': 'trace'}
+[logger.py: 574] (2026-01-16 01:52:33) [DEBUG] Logger: __init__: Logger class custom log levels configured with names: ['FATAL', 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'VERBOSE', 'TRACE'], numbers: {'FATAL': 60, 'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10, 'VERBOSE': 5, 'TRACE': 1}, methods: {'FATAL': 'fatal', 'CRITICAL': 'critical', 'ERROR': 'error', 'WARNING': 'warning', 'INFO': 'info', 'DEBUG': 'debug', 'VERBOSE': 'verbose', 'TRACE': 'trace'}
+[cascade_correlation.py: 317] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: __init__: Initializing Cascade Correlation Network
+[cascade_correlation.py: 346] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_config: Initializing configuration
+[cascade_correlation.py: 350] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_config: Configuration set to: <cascade_correlation.cascade_correlation_config.cascade_correlation_config.CascadeCorrelationConfig object at 0x77b2845c1550>
+[cascade_correlation.py: 3653] (2026-01-16 01:52:33) [TRACE] CascadeCorrelationNetwork: set_uuid: Starting to set UUID for CascadeCorrelationNetwork class
+[cascade_correlation.py: 3654] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: set_uuid: Setting UUID to: None
+[cascade_correlation.py: 3545] (2026-01-16 01:52:33) [TRACE] CascadeCorrelationNetwork: _generate_uuid: Inside the CascadeCorrelationNetwork class Generate UUID method
+[cascade_correlation.py: 3547] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _generate_uuid: UUID: 136484c3-9a30-497d-9254-336944cc05e0
+[cascade_correlation.py: 3548] (2026-01-16 01:52:33) [TRACE] CascadeCorrelationNetwork: _generate_uuid: Completed the CascadeCorrelationNetwork class Generate UUID method
+[cascade_correlation.py: 3661] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: set_uuid: UUID set to: 136484c3-9a30-497d-9254-336944cc05e0
+[cascade_correlation.py: 3662] (2026-01-16 01:52:33) [TRACE] CascadeCorrelationNetwork: set_uuid: Completed setting UUID for CascadeCorrelationNetwork class
+[cascade_correlation.py: 352] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_config: UUID set to: 136484c3-9a30-497d-9254-336944cc05e0
+[cascade_correlation.py: 356] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_logging_system: Initializing logging system
[cascade_correlation.py:382] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_logging_system: Logger initialized with level: 20
+[cascade_correlation.py: 386] (2026-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_network_parameters: Initializing network parameters
[cascade_correlation.py:649] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: 10 with max value: 10 using generator: <bound method RandomState.randint of RandomState(MT19937) at 0x77B28457CE40>
[cascade_correlation.py:649] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: 10 with max value: 10 using generator: <bound method Random.randint of <random.Random object at 0x5760dada83a0>>
[cascade_correlation.py:649] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _roll_sequence_number: Rolling sequence number to: 9 with max value: 10 using generator: <function CascadeCorrelationNetwork._initialize_randomness.<locals>.<lambda> at 0x77b28457a820>
[cascade_correlation.py:682] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_activation_with_derivative: Using activation function: Tanh()
[cascade_correlation.py:437] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_network_parameters: Network parameters initialized
[cascade_correlation.py:484] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_multiprocessing: Multiprocessing components initialized
[cascade_correlation.py:504] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: _init_display_components: Display components initialized
+[cascade_correlation.py: 334] (2026-01-16 01:52:33) [INFO] CascadeCorrelationNetwork: __init__: Initialization completed
[cascade_correlation.py:3680] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: get_uuid: Current UUID: 136484c3-9a30-497d-9254-336944cc05e0
[cascade_correlation.py:3688] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: get_uuid: Returning UUID: 136484c3-9a30-497d-9254-336944cc05e0
<cascade_correlation.cascade_correlation.CascadeCorrelationNetwork object at 0x77b2845c17f0>
[cascade_correlation.py:963] (26-01-16 01:52:33) [INFO] CascadeCorrelationNetwork: fit: Initial training of output layer
[cascade_correlation.py:1090] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Input shape: torch.Size([1600, 2]), Target shape: torch.Size([1600, 2]), Epochs: 50
[cascade_correlation.py:1100] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Input size for output layer: 2, Output size: 2
[cascade_correlation.py:1103] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Adjusted input size for output layer with hidden units: 2
[cascade_correlation.py:1109] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output weights shape: torch.Size([2, 2]), Transposed weights shape: torch.Size([2, 2])
[cascade_correlation.py:1111] (26-01-16 01:52:33) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output bias shape: torch.Size([2]), Bias: Parameter containing:
[cascade_correlation.py:1801] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: _create_optimizer: Created Adam optimizer with lr=0.01
[cascade_correlation.py:1116] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Created optimizer: Adam
[cascade_correlation.py:1118] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Learning Rate: 0.01, Optimizer:
[cascade_correlation.py:1119] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output layer initialized with weights shape: torch.Size([2, 2]), Bias shape: torch.Size([2])
[cascade_correlation.py:1134] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output shape: torch.Size([1600, 2]), Output Input shape: torch.Size([1600, 2])
[cascade_correlation.py:1135] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output:
[cascade_correlation.py:1136] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Target shape: torch.Size([1600, 2]), Target:
[cascade_correlation.py:1145] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch 1, Loss: 1.477029
[cascade_correlation.py:1134] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output shape: torch.Size([1600, 2]), Output Input shape: torch.Size([1600, 2])
[cascade_correlation.py:1135] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output:
[cascade_correlation.py:1136] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Target shape: torch.Size([1600, 2]), Target:
[cascade_correlation.py:1145] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch 2, Loss: 1.327659
[cascade_correlation.py:1134] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output shape: torch.Size([1600, 2]), Output Input shape: torch.Size([1600, 2])
[cascade_correlation.py:1135] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output:
[cascade_correlation.py:1136] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Target shape: torch.Size([1600, 2]), Target:
[cascade_correlation.py:1145] (26-01-16 01:52:34) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Output Layer Training - Epoch 3, Loss: 1.194225
```

Process exited with code 141
`````

The application is running and the training is progressing. Let me see if the manager starts successfully:

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && timeout 180 python src/main.py 2>&1 | grep -E "_start_manager|Manager started|Failed to start manager|correlation=|best_candidate|hidden_units|grow_network" | head -30

```log
[cascade_correlation.py:2404] (26-01-16 01:54:44) [DEBUG] CascadeCorrelationNetwork: grow_network: Epoch 0, Residual Error: 0.099075
[cascade_correlation.py:2022] (26-01-16 01:54:44) [DEBUG] CascadeCorrelationNetwork: _start_manager: Starting multiprocessing manager
[cascade_correlation.py:2039] (26-01-16 01:54:44) [ERROR] CascadeCorrelationNetwork: _start_manager: Failed to start manager:
    self._start_manager()
  File "/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/src/cascade_correlation/cascade_correlation.py", line 2032, in _start_manager
[cascade_correlation.py:1204] (26-01-16 01:54:44) [DEBUG] CascadeCorrelationNetwork: train_candidates: Candidate training results: length: 10, value: [CandidateTrainingResult(candidate_id=0, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=1, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=2, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=Fal
se, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=3, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=4, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=5, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No resu
lts obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=6, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=7, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=8, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Du
mmy Data.'), CandidateTrainingResult(candidate_id=9, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.')]
[cascade_correlation.py:1600] (26-01-16 01:54:44) [DEBUG] CascadeCorrelationNetwork: _process_training_results: Processed results: TrainingResults(epochs_completed=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], candidate_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], candidate_uuids=[], correlations=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], candidate_objects=[], best_candidate_id=0, best_candidate_uuid=None, best_correlation=0.0, best_candidate=None, success_count=0, successful_candidates=0, failed_count=10, error_messages={0: 'Candidate ID 0 (UUID: None): Invalid candidate data', 1: 'Candidate ID 1 (UUID: None): Invalid candidate data', 2: 'Candidate ID 2 (UUID: None): Invalid candidate data', 3: 'Candidate ID 3 (UUID: None): Invalid candidate data', 4: 'Candidate ID 4 (UUID: None): Invalid candidate data', 5: 'Candidate ID 5 (UUID: None): Invalid candidate data', 6: 'Candidate ID 6 (UUID: None): Invalid candidate data', 7: 'Candidate ID 7 (UUID: None): Invalid candidate data', 8: 'Candidate ID 8 (UU
ID: None): Invalid candidate data', 9: 'Candidate ID 9 (UUID: None): Invalid candidate data'}, max_correlation=0.0, start_time=datetime.datetime(2026, 1, 16, 1, 54, 44, 565849), end_time=datetime.datetime(2026, 1, 16, 1, 54, 44, 623611))
[cascade_correlation.py:2408] (26-01-16 01:54:44) [WARNING] CascadeCorrelationNetwork: grow_network: Training results are None or best candidate is None, stopping growth of the network.
[cascade_correlation.py:2469] (26-01-16 01:54:44) [WARNING] CascadeCorrelationNetwork: grow_network: Validation failed at epoch 1/50.
[cascade_correlation.py:2478] (26-01-16 01:54:44) [INFO] CascadeCorrelationNetwork: grow_network: Finished training after 1 epochs. Total hidden units: 0
[cascade_correlation.py:2479] (26-01-16 01:54:44) [DEBUG] CascadeCorrelationNetwork: grow_network: Final history:
{'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': []}
{'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': [{'correlation': 0.0, 'weights': [], 'bias': []}]}
[spiral_problem.py:1196] (26-01-16 01:54:44) [DEBUG] SpiralProblem: solve_n_spiral_problem: Training history: {'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': [{'correlation': 0.0, 'weights': [], 'bias': []}]}
```

Process exited with code 124
`````

The actual error message is empty after "Failed to start manager:". Let me see more context:
The traceback shows the recv() call is failing - this suggests the child process is dying. Let me check stderr for the actual error:

The error is still "Address already in use" even with port 0! This is very strange. Let me investigate by checking if there might be a race condition or if the address is not being dynamically allocated correctly.

Looking at the log, it says "CandidateTrainingManager-2" - suggesting this is the second manager being started (maybe a leftover from a previous test or process). Let me check if there are lingering processes:

No lingering processes. The issue with "CandidateTrainingManager-2" tells me the manager has been instantiated multiple times in the same session.
Let me investigate the possibility that when using fork context and port 0, the OS is assigning the same port to both the parent's socket and the forked child process's socket. This can happen if the socket is created before the fork and the file descriptor is shared.
Actually, thinking about this more - with BaseManager, the server process creates a socket and the client (parent) process connects to it. With port 0, the server should create the socket and get a random available port, then communicate that back to the parent via a pipe.
The error says "Address already in use" in the server process (CandidateTrainingManager-2). The issue might be that the address tuple is being shared incorrectly. Let me look at how BaseManager works - it should handle port 0 correctly.
Actually, looking at the error more carefully - the problem might be that with fork context, the manager server process inherits some state from the parent that already has that address bound. Let me try with spawn context instead:

src/constants/constants_model/ constants_model.py

```python
# _PROJECT_MODEL_AUTHKEY = b'Juniper_Cascade_Correlation_Multiprocessing_Authkey'
# _PROJECT_MODEL_AUTHKEY = 'Juniper_Cascade_Correlation_Multiprocessing_Authkey'
_PROJECT_MODEL_AUTHKEY = "Juniper_Cascade_Correlation_Multiprocessing_Authkey"

# Use 'fork' context on Linux for better compatibility with BaseManager
# Use 'spawn' context for BaseManager to avoid socket inheritance issues with 'fork'
# 'forkserver' has issues with custom Manager classes in Python 3.14
_PROJECT_MODEL_CANDIDATE_TRAINING_CONTEXT = "spawn"

# Define base manager server address and port for remote multiprocessing shared queues
# _PROJECT_MODEL_BASE_MANAGER_ADDRESS = 'localhost'
_PROJECT_MODEL_BASE_MANAGER_ADDRESS_IP = '127.0.0.1'
```

`````bash
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && timeout 120 python src/main.py 2>&1 | grep -E "Manager started|Failed to start manager|Address already|correlation=|hidden|grow" | head -30

```log
[cascade_correlation.py:1103] (26-01-16 02:01:35) [DEBUG] CascadeCorrelationNetwork: train_output_layer: Adjusted input size for output layer with hidden units: 2
[cascade_correlation.py:2404] (26-01-16 02:01:37) [DEBUG] CascadeCorrelationNetwork: grow_network: Epoch 0, Residual Error: 0.099075
OSError: [Errno 98] Address already in use
[cascade_correlation.py:2039] (26-01-16 02:01:39) [ERROR] CascadeCorrelationNetwork: _start_manager: Failed to start manager:
[cascade_correlation.py:1204] (26-01-16 02:01:39) [DEBUG] CascadeCorrelationNetwork: train_candidates: Candidate training results: length: 10, value: [CandidateTrainingResult(candidate_id=0, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=1, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=2, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=Fal
se, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=3, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=4, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=5, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No resu
lts obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=6, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=7, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.'), CandidateTrainingResult(candidate_id=8, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Du
mmy Data.'), CandidateTrainingResult(candidate_id=9, candidate_uuid=None, correlation=0.0, candidate=None, best_corr_idx=None, all_correlations=[], norm_output=None, norm_error=None, numerator=0.0, denominator=1.0, success=False, epochs_completed=0, error_message='No results obtained from candidate training. Using Dummy Data.')]
[cascade_correlation.py:1600] (26-01-16 02:01:39) [DEBUG] CascadeCorrelationNetwork: _process_training_results: Processed results: TrainingResults(epochs_completed=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], candidate_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], candidate_uuids=[], correlations=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], candidate_objects=[], best_candidate_id=0, best_candidate_uuid=None, best_correlation=0.0, best_candidate=None, success_count=0, successful_candidates=0, failed_count=10, error_messages={0: 'Candidate ID 0 (UUID: None): Invalid candidate data', 1: 'Candidate ID 1 (UUID: None): Invalid candidate data', 2: 'Candidate ID 2 (UUID: None): Invalid candidate data', 3: 'Candidate ID 3 (UUID: None): Invalid candidate data', 4: 'Candidate ID 4 (UUID: None): Invalid candidate data', 5: 'Candidate ID 5 (UUID: None): Invalid candidate data', 6: 'Candidate ID 6 (UUID: None): Invalid candidate data', 7: 'Candidate ID 7 (UUID: None): Invalid candidate data', 8: 'Candidate ID 8 (UU
ID: None): Invalid candidate data', 9: 'Candidate ID 9 (UUID: None): Invalid candidate data'}, max_correlation=0.0, start_time=datetime.datetime(2026, 1, 16, 2, 1, 37, 417810), end_time=datetime.datetime(2026, 1, 16, 2, 1, 39, 858078))
[cascade_correlation.py:2408] (26-01-16 02:01:39) [WARNING] CascadeCorrelationNetwork: grow_network: Training results are None or best candidate is None, stopping growth of the network.
[cascade_correlation.py:2469] (26-01-16 02:01:39) [WARNING] CascadeCorrelationNetwork: grow_network: Validation failed at epoch 1/50.
[cascade_correlation.py:2478] (26-01-16 02:01:39) [INFO] CascadeCorrelationNetwork: grow_network: Finished training after 1 epochs. Total hidden units: 0
[cascade_correlation.py:2479] (26-01-16 02:01:39) [DEBUG] CascadeCorrelationNetwork: grow_network: Final history:
{'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': []}
{'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': [{'correlation': 0.0, 'weights': [], 'bias': []}]}
[spiral_problem.py:1196] (26-01-16 02:01:39) [DEBUG] SpiralProblem: solve_n_spiral_problem: Training history: {'train_loss': [0.2347976565361023], 'value_loss': [], 'train_accuracy': [0.615625], 'value_accuracy': [], 'hidden_units_added': [{'correlation': 0.0, 'weights': [], 'bias': []}]}
[cascade_correlation.py:3425] (26-01-16 02:01:39) [INFO] CascadeCorrelationNetwork: summary: Number of hidden units: 0
```

`````

Still the same error. This is bizarre - with port 0, the OS should pick an available port dynamically.

Let me look more closely at how the CandidateTrainingManager is being initialized in _start_manager. I suspect the issue might be that the manager is somehow being created with the same context that already has a socket bound:

####################################################################################################################################################################################################################
# Try changing "fork" to "forkserver". Also, from the documentation "Note that this should be called at most once, and it should be protected inside the if __name__ == '__main__' clause of the main module." – 
jackal
####################################################################################################################################################################################################################


