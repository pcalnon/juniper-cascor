"""Training API request/response models."""

from typing import Any, Dict, List, Optional

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
