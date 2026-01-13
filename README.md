# Juniper: Dynamic Neural Network Research Platform

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms. The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Active Research Components

**juniper_cascor**:  Cascade Correlation Neural Network

- Reference implementation from foundational research (Fahlman & Lebiere, 1990)
- Designed for flexibility, modularity, and scalability
- Enables investigation of constructive learning algorithms

**juniper_canopy**:  Interactive Research Interface

- Research-driven monitoring and visualization environment
- Delivers novel observations through real-time network introspection
- Transforms metrics into insights, accelerating experimental iteration

## Research Philosophy

Juniper prioritizes **transparency over convenience** and **understanding over abstraction**. By implementing algorithms from first principles, the platform provides researchers with increased visibility into network behavior, enabling a more rigorous and more controlled investigation of learning dynamics and architectural innovations.

## Important Notices

### Thread Safety Warning

**The `CascadeCorrelationNetwork` class is NOT thread-safe.** Do not share network instances between threads without proper synchronization. For concurrent training scenarios, create separate network instances per thread. The internal multiprocessing for candidate training is handled within the class and does not require external synchronization.
