# MNIST Hyperparameter Tuning and Experiment Management System

## Overview

This project implements an Experiment Management System using Streamlit for a simple ML/DL task to solve the MNIST challenge with PyTorch. The system provides a user-friendly web interface for hyperparameter tuning, running experiments, and managing the overall experimentation process.

## Key Components

### PyTorch Model

- A PyTorch model (`SimpleNN`) is defined for the MNIST dataset.
- The model architecture includes two fully connected layers.

### Hyperparameter Tuning Sidebar

- Users can tune hyperparameters via the Streamlit sidebar:
  - Learning Rate Slider
  - Epochs Slider
  - Batch Size Slider
  - Hidden Units Slider
  - Optimizer Choice Dropdown

### MLOps Integration with MLflow

- MLflow is used for experiment tracking and logging.
- Hyperparameters, metrics (test accuracy and loss), and the trained model are logged for each experiment.

### Training Process and UI Interaction

- Real-time progress updates are displayed during the training process.
- Users can initiate new experiments or select from existing ones.
- The MLflow UI can be launched directly from the Streamlit app.

### Experiment Management Sidebar

- The Experiment Management sidebar allows users to:
  - Choose New Experiment or Existing Experiment.
  - Input a name for a new experiment.
  - Select from existing experiments.

### Sorting and Resuming

- Experiments can be sorted based on predefined metrics (accuracy and loss) by go to MLFlow UI, choose the desire experiments, choose the Chart tab to see the sorting.
- The UI state is preserved, allowing users to close and reopen the browser without losing the current state.

### Avoiding Duplicate Jobs

- The system checks if the exact same job (same hyperparameters) has already been executed, preventing duplicate runs.

## How to Use

1. Run the Streamlit app using `streamlit run your_script_name.py`.
2. Adjust hyperparameters using the sidebar sliders and options.
3. Choose to start a new experiment or select from existing ones.
4. Click the "Train" button to initiate model training.
5. Monitor the training progress in real-time.
6. Optionally, launch the MLflow UI to explore experiment details.
7. The Experiment Management UI allows sorting and filtering of experiments for ease of comparison.

**Note:** Ensure that all required libraries are installed before running the script. You can install them using `pip install streamlit pandas numpy torch torchvision mlflow`.
