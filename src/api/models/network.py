"""Network API request/response models."""

from typing import Literal

from pydantic import BaseModel, Field

from cascor_constants.constants_api import (
    _PROJECT_API_NETWORK_ACTIVATION_FUNCTION_DEFAULT,
    _PROJECT_API_NETWORK_CANDIDATE_EPOCHS_DEFAULT,
    _PROJECT_API_NETWORK_CANDIDATE_LEARNING_RATE_DEFAULT,
    _PROJECT_API_NETWORK_CANDIDATE_POOL_SIZE_DEFAULT,
    _PROJECT_API_NETWORK_CORRELATION_THRESHOLD_DEFAULT,
    _PROJECT_API_NETWORK_EPOCHS_MAX_DEFAULT,
    _PROJECT_API_NETWORK_INIT_OUTPUT_WEIGHTS_DEFAULT,
    _PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT,
    _PROJECT_API_NETWORK_LEARNING_RATE_DEFAULT,
    _PROJECT_API_NETWORK_MAX_HIDDEN_UNITS_DEFAULT,
    _PROJECT_API_NETWORK_MAX_ITERATIONS_DEFAULT,
    _PROJECT_API_NETWORK_OUTPUT_EPOCHS_DEFAULT,
    _PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT,
    _PROJECT_API_NETWORK_PATIENCE_DEFAULT,
)


class NetworkCreateRequest(BaseModel):
    """Request to create a new CasCor network."""

    input_size: int = Field(_PROJECT_API_NETWORK_INPUT_SIZE_DEFAULT, ge=1, description="Number of input features")
    output_size: int = Field(_PROJECT_API_NETWORK_OUTPUT_SIZE_DEFAULT, ge=1, description="Number of output classes")
    learning_rate: float = Field(_PROJECT_API_NETWORK_LEARNING_RATE_DEFAULT, gt=0, description="Learning rate")
    candidate_learning_rate: float = Field(_PROJECT_API_NETWORK_CANDIDATE_LEARNING_RATE_DEFAULT, gt=0)
    max_hidden_units: int = Field(_PROJECT_API_NETWORK_MAX_HIDDEN_UNITS_DEFAULT, ge=1)
    candidate_pool_size: int = Field(_PROJECT_API_NETWORK_CANDIDATE_POOL_SIZE_DEFAULT, ge=1)
    correlation_threshold: float = Field(_PROJECT_API_NETWORK_CORRELATION_THRESHOLD_DEFAULT, ge=0)
    patience: int = Field(_PROJECT_API_NETWORK_PATIENCE_DEFAULT, ge=1)
    candidate_epochs: int = Field(_PROJECT_API_NETWORK_CANDIDATE_EPOCHS_DEFAULT, ge=1)
    output_epochs: int = Field(_PROJECT_API_NETWORK_OUTPUT_EPOCHS_DEFAULT, ge=1)
    epochs_max: int = Field(_PROJECT_API_NETWORK_EPOCHS_MAX_DEFAULT, ge=1)
    max_iterations: int = Field(_PROJECT_API_NETWORK_MAX_ITERATIONS_DEFAULT, ge=1, description="Maximum cascade growth iterations")
    init_output_weights: Literal["zero", "random"] = Field(_PROJECT_API_NETWORK_INIT_OUTPUT_WEIGHTS_DEFAULT, description="Initialization mode for new hidden unit output weights")
    # CAN-010 / ENH-006 (Phase 6E Sprint A-2): output-layer optimizer.
    # The full registry lives in ``cascade_correlation.py::_create_optimizer``
    # — duplicating it as a Literal here keeps the API surface explicit and
    # gives clients a 422 instead of a runtime warning when they ask for an
    # unsupported optimizer.
    optimizer_type: Literal[
        "Adam",
        "AdamW",
        "SGD",
        "RMSprop",
        "NAdam",
        "RAdam",
        "Adamax",
        "Adagrad",
        "Adadelta",
        "Adafactor",
        "ASGD",
        "LBFGS",
        "Rprop",
        "Muon",
    ] = Field("Adam", description="Output-layer optimizer (defaults to Adam)")
    # CAN-011 (Phase 6E Sprint A-3): output-layer activation function.
    # Mirrors ``_PROJECT_MODEL_ACTIVATION_FUNCTIONS_NAME_LIST`` in
    # ``constants_activation.py`` (the source of truth for the supported
    # registry); duplicating it as a Literal here returns a 422 on
    # unsupported names rather than letting the network silently fall back
    # to the default at ``_init_activation_function`` time.
    activation_function_name: Literal[
        "Identity",
        "Tanh",
        "Sigmoid",
        "ReLU",
        "LeakyReLU",
        "ELU",
        "SELU",
        "GELU",
        "Softmax",
        "Softplus",
        "Hardtanh",
        "Softshrink",
        "Tanhshrink",
        "tanh",
        "sigmoid",
        "relu",
    ] = Field(_PROJECT_API_NETWORK_ACTIVATION_FUNCTION_DEFAULT, description="Hidden-unit activation function (defaults to Tanh)")


class NetworkInfo(BaseModel):
    """Network information response."""

    input_size: int
    output_size: int
    hidden_units: int
    max_hidden_units: int
    learning_rate: float
    uuid: str = ""
