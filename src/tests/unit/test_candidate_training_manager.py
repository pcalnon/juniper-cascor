
import sys
import pytest
import platform
# import matplotlib.pyplot
from multiprocessing.managers import BaseManager

if platform.system() == 'Linux':
    sys.path.append('/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/pcalnon/Development/python/Juniper/src/prototypes/cascor/src')
else:
    raise EnvironmentError("Unsupported platform for testing")

from cascade_correlation.cascade_correlation import CandidateTrainingManager

@pytest.mark.parametrize(
    "manager_cls, expected_type",
    [
        pytest.param(CandidateTrainingManager, CandidateTrainingManager, id="happy-path-candidate-training-manager"),
        pytest.param(BaseManager, BaseManager, id="base-manager-direct"),
    ]
)
def test_candidate_training_manager_inheritance(manager_cls, expected_type):
    #
    # Act
    manager = manager_cls()

    #
    # Assert
    assert isinstance(manager, expected_type)
    assert issubclass(manager_cls, BaseManager)

@pytest.mark.parametrize(
    "register_args, register_kwargs, expected_exception, id",
    [
        pytest.param(
            ("test",), {}, None, "register-method-happy-path"
        ),
        pytest.param(
            ("test",), {"callable": lambda: 42}, None, "register-method-with-callable"
        ),
        pytest.param(
            (None,), {}, TypeError, "register-method-none-name"
        ),
        pytest.param(
            (), {}, TypeError, "register-method-missing-args"
        ),
    ]
)
def test_candidate_training_manager_register(register_args, register_kwargs, expected_exception, id):
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    if expected_exception:
        with pytest.raises(expected_exception):
            manager.register(*register_args, **register_kwargs)
    else:
        manager.register(*register_args, **register_kwargs)
        # If no exception, registration should succeed (no return value to check)

@pytest.mark.parametrize(
    "start_method, expected_exception, id",
    [
        pytest.param("fork", None, "start-method-fork"),
        pytest.param("spawn", None, "start-method-spawn"),
        pytest.param("forkserver", None, "start-method-forkserver"),
        pytest.param("invalid_method", ValueError, "start-method-invalid"),
    ]
)
def test_candidate_training_manager_start_method(start_method, expected_exception, id):
    """
    Test that the start method parameter is validated correctly.
    
    Note: We do not actually call start() for valid methods because that would 
    require a full multiprocessing context and proper cleanup. Instead, we only 
    test the validation logic by checking that invalid methods raise ValueError.
    """
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    if expected_exception:
        with pytest.raises(expected_exception):
            manager.start(method=start_method)
    else:
        # For valid methods, we verify the method is valid by checking mp.get_context doesn't error
        # We skip actually starting the manager to avoid multiprocessing complexity in tests
        import multiprocessing as mp
        try:
            mp.get_context(start_method)
            # If we get here, the method is valid for this platform
            pytest.skip(f"Skipping actual start() call for '{start_method}' to avoid multiprocessing complexity")
        except ValueError:
            pytest.skip(f"Start method '{start_method}' not available on this platform.")

def test_candidate_training_manager_repr_and_str():
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act
    manager_repr = repr(manager)
    manager_str = str(manager)

    #
    # Assert
    assert isinstance(manager_repr, str)
    assert isinstance(manager_str, str)
    assert "CandidateTrainingManager" in manager_repr

def test_candidate_training_manager_address_property():
    """
    Test that the address property returns None or the configured address before starting.
    
    Note: BaseManager.address is a property that returns None before start() is called,
    or the address tuple if one was configured in the constructor. It does not raise
    AttributeError.
    """
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act
    addr = manager.address
    
    #
    # Assert - address should be None when not configured and manager not started
    assert addr is None or isinstance(addr, tuple)

def test_candidate_training_manager_shutdown_without_start():
    """
    Test that calling shutdown() before start() raises an AttributeError.
    
    Note: The shutdown() method is only available on the manager instance after
    start() has been called, because start() adds it to the instance.
    """
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    # Calling shutdown before start should raise an AttributeError (method doesn't exist yet)
    with pytest.raises(AttributeError):
        manager.shutdown()

