# Hessian-Based Methods for Optimization
UCLA CS260D - Large Scale Machine Learning Final Project

## 1. Hessian Approximation
- **Concept**: This method modifies AdaHessian by approximating the Hessian using the diagonal elements. Instead of leveraging the full curvature, it uses a simplified form of the second-order information for faster computation.
- **Difference from AdaHessian**:
  - Simplified diagonal approximation for Hessian.
  - Reduced computational overhead.
- **Contribution**:
  - Offers a balance between computational efficiency and the effectiveness of second-order optimization.
- **Potential Improvements**:
  - Incorporate block-diagonal Hessian approximations for more curvature details without significant computational costs.

## 2. Nonlinear Hessian
- **Concept**: Extends AdaHessian by incorporating a nonlinear transformation (e.g., `tanh`) of the gradient. This allows capturing nonlinearity in the loss surface.
- **Difference from AdaHessian**:
  - Adds a nonlinear factor (`tanh`) to modify the gradient.
  - Attempts to better adapt to non-convex loss surfaces.
- **Contribution**:
  - Handles nonlinearity in optimization, which could improve convergence for complex surfaces.
- **Potential Improvements**:
  - Explore other nonlinear functions or learnable transformations for more flexibility.

## 3. Multi-Scale Hessian
- **Concept**: Combines curvature information at multiple scales. It adjusts learning rates dynamically based on local and global curvature estimates.
- **Difference from AdaHessian**:
  - Incorporates multi-scale Hessian approximations, unlike AdaHessian, which uses a single scale.
- **Contribution**:
  - Balances local adaptability and global convergence for optimization problems.
- **Potential Improvements**:
  - Combine with adaptive learning rates for enhanced efficiency in multi-scale environments.

## 4. Hybrid Hessian-Gradient
- **Concept**: A hybrid method that blends gradient and Hessian-based updates. It dynamically transitions from gradient-based optimization to Hessian-based optimization as training progresses.
- **Difference from AdaHessian**:
  - Uses a weighted combination of gradient and Hessian updates.
  - Tailored for faster convergence in the early stages and higher precision near minima.
- **Contribution**:
  - Provides flexibility by leveraging both first-order and second-order information.
- **Potential Improvements**:
  - Dynamically adjust the weights between gradient and Hessian updates based on training progress.

---

# Code Workflow

## Overview
The provided code automates the training and evaluation of models using multiple optimization methods, including AdaHessian and user-defined second-order optimizers. It saves all results, trained models, and optimizer definitions for reproducibility.

## Workflow
1. **Initialization**:
   - Load CIFAR-10 dataset.
   - Define and initialize the CNN model (`SimpleCNN`).
   - Define AdaHessian and selected user-provided optimizers (`Hessian Approximation`, `Nonlinear Hessian`, etc.).

2. **Training**:
   - For each optimizer (including AdaHessian and selected methods), train the model on CIFAR-10.
   - Track loss and accuracy for each epoch.

3. **Saving Results**:
   - Save training metrics (loss and accuracy) as `.csv` files for each optimizer.
   - Save the trained model weights as `.pth` files for later use.

4. **Comparison**:
   - Plot and save comparison graphs for loss and accuracy across all methods.
   - Save Python files containing the code for each optimizer for reproducibility.

---

# Usage

## How to Use the Code
1. Select the optimization methods you want to run:
   ```python
   selected_methods = ["Hessian Approximation", "Nonlinear Hessian", "Multi-Scale Hessian", "Hybrid Hessian-Gradient"]
   ```

2. Run the main_experiment function:
   ```python
   main_experiment(selected_methods)
   ```
3. Results will be saved in a folder named `experiment_{timestamp}`:

- Trained Models: Stored as `{method_name}_model.pth` files.
- Training Metrics: Stored as `{method_name}.csv` files with Epoch, Loss, and Accuracy.
- Comparison Graphs: Loss and Accuracy comparison graphs as `loss_comparison.png` and `accuracy_comparison.png`.
- Optimizer Code: Optimizer implementations stored as `{method_name}.py`
