"""Network API request/response models."""

from typing import Literal

from pydantic import BaseModel, Field


class NetworkCreateRequest(BaseModel):
    """Request to create a new CasCor network."""

    input_size: int = Field(2, ge=1, description="Number of input features")
    output_size: int = Field(2, ge=1, description="Number of output classes")
    learning_rate: float = Field(0.01, gt=0, description="Learning rate")
    candidate_learning_rate: float = Field(0.005, gt=0)
    max_hidden_units: int = Field(1000, ge=1, description="Maximum hidden units cap (aligned with _PROJECT_MODEL_MAX_HIDDEN_UNITS=1000 and canopy UI)")
    candidate_pool_size: int = Field(8, ge=1)
    correlation_threshold: float = Field(0.1, ge=0)
    patience: int = Field(5, ge=1)
    candidate_epochs: int = Field(50, ge=1)
    output_epochs: int = Field(25, ge=1)
    epochs_max: int = Field(1000000, ge=1, description="Global maximum training epochs (aligned with canopy nn_max_total_epochs default of 1,000,000)")
    max_iterations: int = Field(1000, ge=1, description="Maximum cascade growth iterations")
    init_output_weights: Literal["zero", "random"] = Field("zero", description="Initialization mode for new hidden unit output weights")


class NetworkInfo(BaseModel):
    """Network information response."""

    input_size: int
    output_size: int
    hidden_units: int
    max_hidden_units: int
    learning_rate: float
    uuid: str = ""
