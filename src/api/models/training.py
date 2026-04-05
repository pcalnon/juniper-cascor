"""Training API request/response models."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetSource(BaseModel):
    """Dataset source specification for training."""

    source: str = Field("inline", description="Dataset source: 'inline' or 'juniper-data'")
    url: Optional[str] = Field(None, description="URL for juniper-data source")
    generator: Optional[str] = Field(None, description="Generator name (e.g., 'spiral')")
    params: Optional[Dict[str, Any]] = Field(None, description="Generator parameters")


class InlineDataset(BaseModel):
    """Inline dataset provided directly in the request body."""

    train_x: List[List[float]] = Field(..., description="Training features (2D array)")
    train_y: List[List[float]] = Field(..., description="Training targets (2D array)")
    val_x: Optional[List[List[float]]] = Field(None, description="Validation features")
    val_y: Optional[List[List[float]]] = Field(None, description="Validation targets")


class TrainingStartRequest(BaseModel):
    """Request to start training."""

    epochs: Optional[int] = Field(None, ge=1, description="Max epochs override")
    dataset: Optional[DatasetSource] = Field(None, description="Dataset source specification")
    inline_data: Optional[InlineDataset] = Field(None, description="Inline dataset")
    params: Optional[Dict[str, Any]] = Field(None, description="Training params (learning_rate, patience, etc.)")


class TrainingStatus(BaseModel):
    """Training status response."""

    training_active: bool
    network_loaded: bool
    state_machine: dict
    monitor: dict
    training_state: dict


class TrainingParamUpdateRequest(BaseModel):
    """Runtime-modifiable training parameters (PATCH semantics — all fields optional)."""

    learning_rate: Optional[float] = Field(None, gt=0, description="Output layer learning rate")
    candidate_learning_rate: Optional[float] = Field(None, gt=0, description="Candidate training learning rate")
    correlation_threshold: Optional[float] = Field(None, gt=0, le=1.0, description="Minimum correlation to accept candidate")
    candidate_pool_size: Optional[int] = Field(None, ge=1, description="Number of candidate units per round")
    max_hidden_units: Optional[int] = Field(None, ge=1, description="Maximum hidden units (takes effect on next cascade)")
    epochs_max: Optional[int] = Field(None, ge=1, description="Global maximum training epochs")
    max_iterations: Optional[int] = Field(None, ge=1, description="Maximum cascade growth iterations")
    patience: Optional[int] = Field(None, ge=1, description="Early stopping patience epochs")
    convergence_threshold: Optional[float] = Field(None, gt=0, description="Minimum loss improvement to reset patience counter")
    candidate_convergence_threshold: Optional[float] = Field(None, gt=0, description="Minimum loss improvement for candidate training patience")
    candidate_patience: Optional[int] = Field(None, ge=1, description="Candidate training early stopping patience epochs")
    candidate_epochs: Optional[int] = Field(None, ge=1, description="Number of epochs for candidate training")
    init_output_weights: Optional[Literal["zero", "random"]] = Field(None, description="Initialization mode for new hidden unit output weights")
