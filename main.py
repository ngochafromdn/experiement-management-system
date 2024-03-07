# Import Libraries

import streamlit as st  
import pandas as pd
import numpy as np
import subprocess
import os
import webbrowser


# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# MLOPS
import mlflow

# PyTorch Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configure Page
st.set_page_config(
    page_title="MNIST Hyperparameter Tuning",
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded") 
st.markdown("<h1 style='text-align: center; color: #ff6347;'>MNIST Hyperparameter Tuning 🤖</h1>", unsafe_allow_html=True)


# Define transformation for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
mnist_train = datasets.MNIST('./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Define data loader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# HELPER FUNCTIONS

# Train the model
def train_model(exp_name, model, optimizer, criterion, epochs):
    # Create or Select Experiment 
    experiment = mlflow.set_experiment(exp_name)    
    mlflow.end_run()
    with mlflow.start_run(experiment_id=experiment.experiment_id):          
        # Training loop
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            
            
            # Evaluate on test set
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_accuracy = correct / total
            loss = loss.item()
            

                
            # Log Parameters & Metrics
            mlflow.log_params({"learning_rate": optimizer.param_groups[0]['lr'], "epochs": epochs, "batch_size": train_loader.batch_size, 
                              "hidden_units": model.fc1.out_features, "optimizer": optimizer_choice})     
            mlflow.log_metrics({"Test Accuracy": test_accuracy})
            mlflow.log_metrics({"Loss": loss})
            
            # Log Model
            mlflow.pytorch.log_model(model, "model")
            # Print current running progress on Streamlit
            # Style these words with red color to make them more visible            
            st.write(f"<span style='color: #ff0001;'>New running experiment:</span> {exp_name}", unsafe_allow_html=True)
            st.write(f"Epoch: {epoch+1}/{epochs}, Test Accuracy: {test_accuracy:.3f}, Loss: {loss:.3f}")
            


            
    return test_accuracy, loss

# Function for opening MLFlow UI directly from Streamlit
def open_mlflow_ui():
    # Start the MLflow tracking server as a subprocess
    cmd = "mlflow ui --port 5000"
    subprocess.Popen(cmd, shell=True)
    
def open_browser(url):
    webbrowser.open_new_tab(url)



# STREAMLIT UI   


# Hyperparameter Tuning Sidebar
st.sidebar.title("Hyperparameter Tuning ⚙️")
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.001, max_value=0.1, step=0.001, value=0.01, key="learning_rate")
epochs = st.sidebar.slider('Epochs', min_value=5, max_value=50, step=5, value=20, key="epochs")
batch_size = st.sidebar.slider('Batch Size', min_value=32, max_value=128, step=32, value=64, key="batch_size")
hidden_units = st.sidebar.slider('Hidden Units', min_value=64, max_value=256, step=64, value=128, key="hidden_units")
optimizer_choice = st.sidebar.selectbox('Optimizer', ['SGD', 'Adam', 'RMSprop'], index=1, key="optimizer_choice")


    
# Launch Mlflow from Streamlit
st.sidebar.title("Mlflow Tracking 🔎")    
if st.sidebar.button("See all created experiments (running and completed) 🚀"):
    open_mlflow_ui()
    st.sidebar.success("CLICK TO SEE PROGRESS AND RESULTS! http://localhost:5000")
    open_browser("http://localhost:5000")
def run_exists(params):
    if len(param_set) == 0:
        return False
    for p in param_set:
        if p == params:
            return True
    return False

param_set = []

# Main Page Content
exp_type = st.radio("Select Experiment Type", ['New Experiment', 'Existing Experiment'], horizontal=True)
if exp_type == 'New Experiment':
    exp_name = st.text_input("Enter the name for New Experiment")
    # Training the model starts from here    
    if st.button("Train this new experiment ⚙️", key = "train_button"):
        with st.spinner('Training the model... 🧠'):
            st.write("Training for the new experiment started!")
            #attach link to ML flow button inside these words
            st.write(f"Click on the Mlflow Tracking button on the left to see the progress and results of the experiment: {exp_name}")
        if run_exists({learning_rate, epochs, batch_size, hidden_units, optimizer_choice}):
                st.warning("The exact same job has already been run. Skipping.")
        else:
                # Start a new run
                model = SimpleNN()
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                test_acc = train_model(exp_name, model, optimizer, criterion, epochs)

else:
    try:
        if os.path.exists('./mlruns'):
            exps = [i.name for i in mlflow.search_experiments()]
            exp_name = st.selectbox("Select Experiment", exps)
            st.write(f"Click on the Mlflow button on the left to see the progress and results of the experiment: {exp_name}")
            #hidden source to the mlflow ui page of them selected experiment
            if st.button("See this experiment's progress and results"):
                open_browser(f"http://localhost:5000/#/experiments/{exp_name}")


        else:
            st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")            
    except:
        st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
