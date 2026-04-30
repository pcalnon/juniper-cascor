"""Training API request/response models."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cascor_constants.constants_api import _PROJECT_API_DATASET_SOURCE_DEFAULT, _PROJECT_API_MAX_DATASET_SAMPLES, _PROJECT_API_MAX_DATASET_TARGETS


class DatasetSource(BaseModel):
    """Dataset source specification for training."""

    source: str = Field(_PROJECT_API_DATASET_SOURCE_DEFAULT, description="Dataset source: 'inline' or 'juniper-data'")
    url: Optional[str] = Field(None, description="URL for juniper-data source")
    generator: Optional[str] = Field(None, description="Generator name (e.g., 'spiral')")
    params: Optional[Dict[str, Any]] = Field(None, description="Generator parameters")


class InlineDataset(BaseModel):
    """Inline dataset provided directly in the request body.

    Size limits enforce that inline data is for small ad-hoc datasets only.
    For large datasets, use the juniper-data service.
    """

    train_x: List[List[float]] = Field(..., max_length=_PROJECT_API_MAX_DATASET_SAMPLES, description="Training features (2D array)")
    train_y: List[List[float]] = Field(..., max_length=_PROJECT_API_MAX_DATASET_TARGETS, description="Training targets (2D array)")
    val_x: Optional[List[List[float]]] = Field(None, max_length=_PROJECT_API_MAX_DATASET_SAMPLES, description="Validation features")
    val_y: Optional[List[List[float]]] = Field(None, max_length=_PROJECT_API_MAX_DATASET_TARGETS, description="Validation targets")


class TrainingParams(BaseModel):
    """Validated training parameter overrides for ``POST /training/start`` (SEC-07).

    Each field has the same type + range constraints as the runtime-modifiable
    ``TrainingParamUpdateRequest`` so the start-of-training validation cannot
    diverge from the patch validation. ``extra="forbid"`` ensures unknown
    keys raise 422 instead of being silently dropped — closing the gap that
    the previous ``Dict[str, Any]`` + whitelist-in-handler pattern left open
    (values were never range-checked and allow-listed keys were only
    enforced server-side).
    """

    model_config = ConfigDict(extra="forbid")

    max_epochs: Optional[int] = Field(None, ge=1, description="Max epochs override")
    max_iterations: Optional[int] = Field(None, ge=1, description="Maximum cascade growth iterations")
    early_stopping: Optional[bool] = Field(None, description="Whether early stopping is enabled")
    learning_rate: Optional[float] = Field(None, gt=0, le=10.0, description="Output layer learning rate")
    candidate_learning_rate: Optional[float] = Field(None, gt=0, le=10.0, description="Candidate training learning rate")
    correlation_threshold: Optional[float] = Field(None, gt=0, le=1.0, description="Minimum correlation to accept candidate")
    candidate_pool_size: Optional[int] = Field(None, ge=1, le=256, description="Number of candidate units per round")
    max_hidden_units: Optional[int] = Field(None, ge=1, le=10_000, description="Maximum hidden units")
    epochs_max: Optional[int] = Field(None, ge=1, le=1_000_000, description="Global maximum training epochs")
    patience: Optional[int] = Field(None, ge=1, le=100_000, description="Early stopping patience epochs")
    convergence_threshold: Optional[float] = Field(None, gt=0, description="Minimum loss improvement to reset patience counter")
    candidate_convergence_threshold: Optional[float] = Field(None, gt=0, description="Minimum loss improvement for candidate training patience")
    candidate_patience: Optional[int] = Field(None, ge=1, le=100_000, description="Candidate training patience epochs")
    candidate_epochs: Optional[int] = Field(None, ge=1, le=1_000_000, description="Number of epochs for candidate training")
    # CAS-002 (Phase 6E Sprint A-1): per-output-training-phase epoch budget,
    # distinct from the global ``epochs_max``. The network already exposes
    # ``self.output_epochs`` and consumes ``self.config.output_epochs`` at
    # construction; this surfaces it on the start-of-training override surface.
    output_epochs: Optional[int] = Field(None, ge=1, le=1_000_000, description="Per-output-training-phase epoch budget (separate from epochs_max)")
    init_output_weights: Optional[Literal["zero", "random"]] = Field(None, description="Initialization mode for new hidden unit output weights")


class TrainingStartRequest(BaseModel):
    """Request to start training."""

    epochs: Optional[int] = Field(None, ge=1, description="Max epochs override")
    dataset: Optional[DatasetSource] = Field(None, description="Dataset source specification")
    inline_data: Optional[InlineDataset] = Field(None, description="Inline dataset")
    params: Optional[TrainingParams] = Field(None, description="Training params (learning_rate, patience, etc.)")


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
    # CAS-002 (Phase 6E Sprint A-1): runtime-patchable counterpart to TrainingParams.output_epochs.
    output_epochs: Optional[int] = Field(None, ge=1, description="Per-output-training-phase epoch budget (separate from epochs_max)")
    init_output_weights: Optional[Literal["zero", "random"]] = Field(None, description="Initialization mode for new hidden unit output weights")
