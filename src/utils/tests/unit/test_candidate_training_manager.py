
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
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    if expected_exception:
        with pytest.raises(expected_exception):
            manager.start(method=start_method)
    else:
        # The start method may not be available on all platforms, so we check for NotImplementedError as well
        try:
            manager.start(method=start_method)
        except NotImplementedError:
            pytest.skip(f"Start method '{start_method}' not implemented on this platform.")

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
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    # The address property is not set until the manager is started, so it should raise an AttributeError
    with pytest.raises(AttributeError):
        _ = manager.address

def test_candidate_training_manager_shutdown_without_start():
    #
    # Arrange
    manager = CandidateTrainingManager()

    #
    # Act & Assert
    # Calling shutdown before start should raise an AssertionError
    with pytest.raises(AssertionError):
        manager.shutdown()

