#!/usr/bin/env python
"""
Project:       Juniper
Prototype:     Cascade Correlation Neural Network  
File Name:     generators.py
Author:        Paul Calnon
Version:       0.1.0

Date:          2025-09-26
Last Modified: 2025-09-26

License:       MIT License
Copyright:     Copyright (c) 2024-2025 Paul Calnon

Description:
    Test data generators for Cascade Correlation Network testing.
    Provides various synthetic datasets for comprehensive testing.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Information about a generated dataset."""
    name: str
    input_size: int
    output_size: int
    n_samples: int
    difficulty: str
    description: str


class SpiralDataGenerator:
    """Generator for N-spiral classification problems."""
    
    @staticmethod
    def generate_2_spiral(
        n_per_spiral: int = 100,
        noise: float = 0.1,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate 2-spiral classification problem.
        
        Args:
            n_per_spiral: Number of points per spiral
            noise: Amount of noise to add
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        t = torch.linspace(0.5, 4*np.pi, n_per_spiral)
        
        # Spiral 1
        x1 = t * torch.cos(t) / (4*np.pi) + noise * torch.randn(n_per_spiral)
        y1 = t * torch.sin(t) / (4*np.pi) + noise * torch.randn(n_per_spiral)
        
        # Spiral 2 (rotated 180 degrees)
        x2 = -t * torch.cos(t) / (4*np.pi) + noise * torch.randn(n_per_spiral)
        y2 = -t * torch.sin(t) / (4*np.pi) + noise * torch.randn(n_per_spiral)
        
        x = torch.stack([
            torch.cat([x1, x2]),
            torch.cat([y1, y2])
        ], dim=1)
        
        y = torch.cat([
            torch.tensor([[1, 0]] * n_per_spiral),
            torch.tensor([[0, 1]] * n_per_spiral)
        ], dim=0).float()
        
        info = DatasetInfo(
            name="2-spiral",
            input_size=2,
            output_size=2,
            n_samples=2 * n_per_spiral,
            difficulty="hard",
            description="Classic 2-spiral classification problem"
        )
        
        return x, y, info
    
    @staticmethod
    def generate_n_spiral(
        n_spirals: int = 3,
        n_per_spiral: int = 50,
        noise: float = 0.05,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate N-spiral classification problem.
        
        Args:
            n_spirals: Number of spirals/classes
            n_per_spiral: Number of points per spiral
            noise: Amount of noise to add
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        x_data = []
        y_data = []
        
        for i in range(n_spirals):
            t = torch.linspace(0.5, 4*np.pi, n_per_spiral)
            angle_offset = 2 * np.pi * i / n_spirals
            
            x_spiral = t * torch.cos(t + angle_offset) / (4*np.pi) + noise * torch.randn(n_per_spiral)
            y_spiral = t * torch.sin(t + angle_offset) / (4*np.pi) + noise * torch.randn(n_per_spiral)
            
            x_data.append(torch.stack([x_spiral, y_spiral], dim=1))
            
            # One-hot encoding for class i
            y_spiral = torch.zeros(n_per_spiral, n_spirals)
            y_spiral[:, i] = 1
            y_data.append(y_spiral)
        
        x = torch.cat(x_data, dim=0)
        y = torch.cat(y_data, dim=0)
        
        info = DatasetInfo(
            name=f"{n_spirals}-spiral",
            input_size=2,
            output_size=n_spirals,
            n_samples=n_spirals * n_per_spiral,
            difficulty="very_hard" if n_spirals > 3 else "hard",
            description=f"{n_spirals}-spiral classification problem"
        )
        
        return x, y, info


class ClassificationDataGenerator:
    """Generator for various classification problems."""
    
    @staticmethod
    def generate_xor(
        n_per_class: int = 100,
        noise: float = 0.1,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate XOR problem data.
        
        Args:
            n_per_class: Number of points per class
            noise: Amount of noise to add
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)

        # XOR pattern: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        centers = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 1, 0])

        x_data = []
        y_data = []

        # for i, (center, label) in enumerate(zip(centers, labels)):
        for center, label in zip(centers, labels, strict=False):
            # Generate points around each center
            points = center.unsqueeze(0) + noise * torch.randn(n_per_class, 2)
            x_data.append(points)

            # One-hot encoding
            y_class = torch.zeros(n_per_class, 2)
            y_class[:, label] = 1
            y_data.append(y_class)

        x = torch.cat(x_data, dim=0)
        y = torch.cat(y_data, dim=0)

        info = DatasetInfo(
            name="xor",
            input_size=2,
            output_size=2,
            n_samples=4 * n_per_class,
            difficulty="medium",
            description="XOR classification problem"
        )

        return x, y, info
    
    @staticmethod
    def generate_circles(
        n_samples: int = 200,
        noise: float = 0.1,
        inner_radius: float = 0.3,
        outer_radius: float = 0.7,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate concentric circles classification problem.
        
        Args:
            n_samples: Total number of samples
            noise: Amount of noise to add
            inner_radius: Radius of inner circle
            outer_radius: Radius of outer circle
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        n_inner = n_samples // 2
        n_outer = n_samples - n_inner
        
        # Inner circle
        angles_inner = 2 * np.pi * torch.rand(n_inner)
        r_inner = inner_radius + noise * torch.randn(n_inner)
        x_inner = r_inner * torch.cos(angles_inner)
        y_inner = r_inner * torch.sin(angles_inner)
        
        # Outer circle  
        angles_outer = 2 * np.pi * torch.rand(n_outer)
        r_outer = outer_radius + noise * torch.randn(n_outer)
        x_outer = r_outer * torch.cos(angles_outer)
        y_outer = r_outer * torch.sin(angles_outer)
        
        x = torch.stack([
            torch.cat([x_inner, x_outer]),
            torch.cat([y_inner, y_outer])
        ], dim=1)
        
        y = torch.cat([
            torch.tensor([[1, 0]] * n_inner),
            torch.tensor([[0, 1]] * n_outer)
        ], dim=0).float()
        
        info = DatasetInfo(
            name="circles",
            input_size=2,
            output_size=2,
            n_samples=n_samples,
            difficulty="medium",
            description="Concentric circles classification problem"
        )
        
        return x, y, info


class RegressionDataGenerator:
    """Generator for regression problems."""
    
    @staticmethod
    def generate_polynomial(
        n_samples: int = 200,
        degree: int = 3,
        noise: float = 0.1,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate polynomial regression problem.
        
        Args:
            n_samples: Number of samples
            degree: Polynomial degree
            noise: Amount of noise to add
            x_range: Range for input values
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x = torch.linspace(x_range[0], x_range[1], n_samples).unsqueeze(1)
        
        # Generate polynomial coefficients
        coeffs = torch.randn(degree + 1)
        
        # Compute polynomial
        y = torch.zeros(n_samples, 1)
        for i, coeff in enumerate(coeffs):
            y += coeff * (x ** i)
        
        # Add noise
        y += noise * torch.randn_like(y)
        
        info = DatasetInfo(
            name=f"polynomial_deg_{degree}",
            input_size=1,
            output_size=1,
            n_samples=n_samples,
            difficulty="easy" if degree <= 2 else "medium",
            description=f"Polynomial regression of degree {degree}"
        )
        
        return x, y, info
    
    @staticmethod
    def generate_sinusoidal(
        n_samples: int = 200,
        frequency: float = 1.0,
        noise: float = 0.1,
        x_range: Tuple[float, float] = (0.0, 4*np.pi),
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate sinusoidal regression problem.
        
        Args:
            n_samples: Number of samples
            frequency: Frequency of sine wave
            noise: Amount of noise to add
            x_range: Range for input values
            seed: Random seed for reproducibility
            
        Returns:
            (x, y, info): Input data, targets, and dataset info
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x = torch.linspace(x_range[0], x_range[1], n_samples).unsqueeze(1)
        y = torch.sin(frequency * x) + noise * torch.randn_like(x)
        
        info = DatasetInfo(
            name=f"sinusoidal_freq_{frequency}",
            input_size=1,
            output_size=1,
            n_samples=n_samples,
            difficulty="medium",
            description=f"Sinusoidal regression with frequency {frequency}"
        )
        
        return x, y, info


class CorrelationTestDataGenerator:
    """Generator for testing correlation calculations."""
    
    @staticmethod
    def generate_perfect_correlation(
        n_samples: int = 100,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data with perfect positive correlation.
        
        Returns:
            (x, y): Data with correlation coefficient = 1.0
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x = torch.randn(n_samples, 1)
        y = 2 * x + 1  # Perfect linear relationship
        
        return x, y
    
    @staticmethod
    def generate_no_correlation(
        n_samples: int = 100,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data with no correlation.
        
        Returns:
            (x, y): Data with correlation coefficient ≈ 0.0
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x = torch.randn(n_samples, 1)
        y = torch.randn(n_samples, 1)  # Independent random data
        
        return x, y
    
    @staticmethod
    def generate_negative_correlation(
        n_samples: int = 100,
        seed: Optional[int] = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data with perfect negative correlation.
        
        Returns:
            (x, y): Data with correlation coefficient = -1.0
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        x = torch.randn(n_samples, 1)
        y = -2 * x + 1  # Perfect negative linear relationship
        
        return x, y


def get_test_datasets() -> List[str]:
    """Get list of available test dataset names."""
    return [
        "2-spiral",
        "3-spiral",
        "4-spiral",
        "xor", 
        "circles",
        "polynomial_deg_2",
        "polynomial_deg_3",
        "sinusoidal_freq_1",
        "perfect_correlation",
        "no_correlation",
        "negative_correlation"
    ]


def generate_dataset(name: str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Optional[DatasetInfo]]:
    """
    Generate a dataset by name.
    
    Args:
        name: Dataset name
        **kwargs: Additional parameters for dataset generation
        
    Returns:
        (x, y, info): Input data, targets, and dataset info (if available)
    """
    generators = {
        "2-spiral": lambda: SpiralDataGenerator.generate_2_spiral(**kwargs),
        "3-spiral": lambda: SpiralDataGenerator.generate_n_spiral(n_spirals=3, **kwargs),
        "4-spiral": lambda: SpiralDataGenerator.generate_n_spiral(n_spirals=4, **kwargs),
        "xor": lambda: ClassificationDataGenerator.generate_xor(**kwargs),
        "circles": lambda: ClassificationDataGenerator.generate_circles(**kwargs),
        "polynomial_deg_2": lambda: RegressionDataGenerator.generate_polynomial(degree=2, **kwargs),
        "polynomial_deg_3": lambda: RegressionDataGenerator.generate_polynomial(degree=3, **kwargs),
        "sinusoidal_freq_1": lambda: RegressionDataGenerator.generate_sinusoidal(frequency=1.0, **kwargs),
        "perfect_correlation": lambda: (*CorrelationTestDataGenerator.generate_perfect_correlation(**kwargs), None),
        "no_correlation": lambda: (*CorrelationTestDataGenerator.generate_no_correlation(**kwargs), None),
        "negative_correlation": lambda: (*CorrelationTestDataGenerator.generate_negative_correlation(**kwargs), None),
    }
    
    if name not in generators:
        raise ValueError(f"Unknown dataset name: {name}. Available: {list(generators.keys())}")
    
    return generators[name]()
