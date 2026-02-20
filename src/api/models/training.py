"""Training API request/response models."""

from typing import Optional

from pydantic import BaseModel, Field


class TrainingStartRequest(BaseModel):
    """Request to start training."""

    epochs: Optional[int] = Field(None, ge=1, description="Max epochs override")


class TrainingStatus(BaseModel):
    """Training status response."""

    training_active: bool
    network_loaded: bool
    state_machine: dict
    monitor: dict
    training_state: dict
