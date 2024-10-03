#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import subprocess
import mlflow
from mlflow import MlflowClient

# Sidebar with tabs
st.sidebar.title("Model Workflow")
tab = st.sidebar.radio("Select an action", ["Optimize Models", "Train Final Model", "Make Decision"])

# Variables for uploaded file and execution control
uploaded_file = None
optimization_done = False

############################
def list_experiments_and_get_last_run_id(run_name="collect_params_logging"):
    """
    List all experiments and get the run ID of the last run with the specified run name.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ACTIVE_ONLY)

    for experiment in experiments:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
        for run in runs:
            if run.data.tags.get('mlflow.runName') == run_name:
                return run.info.run_id
    mlflow.end_run()
    return None

def load_dataframe_from_mlflow_artifact(run_id, artifact_path="data_frames/res_collect_params.csv"):
    """
    Load a DataFrame from an MLflow artifact.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    df = pd.read_csv(local_dir)
    mlflow.end_run()
    return df

def run_compilation():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    run_id = list_experiments_and_get_last_run_id(run_name="collect_params_logging")
    df_from_mlflow = load_dataframe_from_mlflow_artifact(run_id)
    mlflow.end_run()
    return df_from_mlflow

################################

# Function to run optimization scripts
def run_optimizations(uploaded_file_path):
    # Run the optimization scripts with the uploaded file path
    subprocess.run(["python", "optimize_metric_1.py", uploaded_file_path])
    subprocess.run(["python", "optimize_metric_2.py", uploaded_file_path])
    st.success("Optimization Complete! Job finished.")

def run_perfcompilation():
    subprocess.run(["python", "save_mlflow_perfcompilations.py"])
    st.success("Performance compilation Complete! Job finished.")

def train_final_model(best_model, uploaded_file_path):
    # Run the final model training scripts with the required arguments
    subprocess.run(["python", "final_model_metric_1.py", "--best_model", best_model, "--uploaded_file", uploaded_file_path])
    subprocess.run(["python", "final_model_metric_2.py", "--best_model", best_model, "--uploaded_file", uploaded_file_path])
    st.success("Final Model Training Complete! Both metrics have been processed.")


# Tab 1: Optimize Models
if tab == "Optimize Models":
    st.header("Optimize Models")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary path and use it for subsequent processing
        with open("/tmp/uploaded_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded file content
        df = pd.read_csv("/tmp/uploaded_file.csv")
        st.dataframe(df)

        # Run optimization button
        if st.button("Run Optimization"):
            with st.spinner('Running optimizations...'):
                run_optimizations("/tmp/uploaded_file.csv")
                run_perfcompilation()
                
            st.warning("Optimization is complete! Click 'OK' to dismiss.")
            if st.button("OK"):
                st.experimental_rerun()
            st.header("Performance compilation")
            st.dataframe(run_compilation())

# Tab 2: Train Final Model
elif tab == "Train Final Model":
    st.header("Train Final Model")
    
    # Dropdown to select the best model
    model_choices = [
        "LogisticRegression_F10", 
        "LogisticRegression_ROC_AUC", 
        "GradientBoosting_F10", 
        "GradientBoosting_ROC_AUC", 
        "NeuralNetwork_F10", 
        "NeuralNetwork_ROC_AUC"
    ]
    
    best_model = st.selectbox("Select the Best Model", model_choices)

    st.write(f"You selected the model: {best_model}")
    uploaded_file = "/tmp/uploaded_file.csv"
    if uploaded_file is None:
        st.warning("Please upload a CSV file in the Optimize Models tab before training the final model.")
    else:
        # Button to trigger final model training
        if st.button("Run Final Model"):
            with st.spinner('Training final model...'):
                train_final_model(best_model, uploaded_file)

# Tab 3: Make Decision
elif tab == "Make Decision":
    st.header("Make Decision")
    st.write("This section will provide tools for making a final decision based on the model performance.")
