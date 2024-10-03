#!/usr/bin/env python3

import mlflow
import pandas as pd
import joblib
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI globally
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def register_and_promote_model(best_model, model_name, artifact_path, run_id, stage="Production"):
    """
    Enregistre et promeut un modèle dans le Model Registry de MLflow.

    Parameters:
    best_model: Le modèle à enregistrer
    model_name: Nom du modèle pour le Model Registry
    artifact_path: Le chemin de l'artefact du modèle (ex: "final_model.joblib")
    run_id: ID du run en cours dans MLflow
    stage: Le stage dans lequel promouvoir le modèle (default: "Production")
    """

    # Save the model as an artifact
    joblib.dump(best_model, artifact_path)

    # Log the artifact in MLflow
    mlflow.log_artifact(artifact_path)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    try:
        # Register the model in MLflow
        registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Modèle {model_name} enregistré.")
        
        # Get the latest version of the model
        latest_version_info = client.get_latest_versions(name=model_name, stages=["None"])
        
        if latest_version_info:
            latest_version = latest_version_info[0].version

            # Promote the model to the desired stage (e.g., Production)
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage
            )
            print(f"Modèle {model_name} version {latest_version} promu en {stage}.")
            
    except Exception as e:
        print(f"Erreur lors de l'enregistrement ou de la promotion du modèle: {e}")

def collect_mlflow_model_hyperparams():
    # Step 1: Connect to MLflow server
    try:
        # Step 2: Search for all experiments
        experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        print("Connexion au serveur MLflow. Expériences détectées:")
        for exp in experiments:
            print(f"Expérience ID: {exp.experiment_id}, Nom: {exp.name}")
    except Exception as e:
        print("Échec de la connexion au serveur MLflow:", e)
        return

    # Step 3: Use the default experiment (ID = 0)
    experiment_id = "0"

    # Step 4: Search for runs in the selected experiment
    runs = client.search_runs(experiment_ids=[experiment_id])

    # Step 5: Initialize lists to store the hyperparameters (params) for each run
    Runs = []
    ParamName = []
    ParamValue = []

    # Step 6: Loop through each run and collect hyperparameters
    for run in runs:
        params = run.data.params
        for param_name, param_value in params.items():
            Runs.append(run.data.tags.get('mlflow.runName'))
            ParamName.append(param_name)
            ParamValue.append(param_value)

    # Step 7: Create a DataFrame with the collected hyperparameters
    hyperparams_df = pd.DataFrame({"Run": Runs, "ParamName": ParamName, "ParamValue": ParamValue})
    
    # Optional: Sort or filter based on hyperparameters, if needed
    sorted_hyperparams_df = hyperparams_df.sort_values(by=["Run", "ParamName"])

    return sorted_hyperparams_df

def collect_mlflow_model_perfs():
    import mlflow
    from mlflow.tracking import MlflowClient
    import os
    import pandas as pd
    import numpy as np

    
    try:
        experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        print("Connexion au serveur MLflow. Expériences détectées:")
        for exp in experiments:
            print(f"Expérience ID: {exp.experiment_id}, Nom: {exp.name}")
    except Exception as e:
        print("Échec de la connexion au serveur MLflow:", e)
    
    # Utiliser l'experiment par défaut avec ID = 0
    experiment_id = "0"
    
    # Récupérer les runs pour l'experiment par défaut
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    # Afficher les noms des runs
    run_names = [run.data.tags.get("mlflow.runName") for run in runs]
    for idx, run_name in enumerate(run_names):
        print(f"Run {idx+1}: {run_name}")
    
    # Pour chaque run, lire les métriques loggées avec arrondi à 2 chiffres
    Runs=[]
    MetricName=[]
    MetricValue=[]
    for run in runs:
        #print(f"Run ID: {run.info.run_id}")
        #print(f"Run Name: {run.data.tags.get('mlflow.runName')}")
        
        # Afficher toutes les métriques loggées pour ce run
        metrics = run.data.metrics
        for metric_name, metric_value in metrics.items():
            #print(f"{metric_name}: {round(metric_value, 2)}")
            Runs.append(run.data.tags.get('mlflow.runName'))
            MetricName.append(metric_name)
            MetricValue.append(metric_value)
        
        #print("-" * 40)
    
    
    
    compilation = pd.DataFrame({"Runs":Runs,"MetricName":MetricName,"MetricValue":MetricValue})
    best_model_per_metric= compilation.groupby("MetricName").apply(lambda x: x[["Runs","MetricValue"]][x["MetricValue"]==np.max(x["MetricValue"])] )
    best_model_per_metric["MetricName"]=[i[0] for i in best_model_per_metric.index]
    best_model_per_metric_test=best_model_per_metric.loc[best_model_per_metric.MetricName.str.contains("test"),:]
    best_model_per_metric_test.sort_values("Runs")
    return best_model_per_metric_test


def log_dataframe_to_mlflow(df, experiment_name="compilation_perfs"):
    try:
        # Ensure MLflow is tracking properly by listing experiments
        experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        print("Connexion au serveur MLflow. Expériences détectées:")
        for exp in experiments:
            print(f"Expérience ID: {exp.experiment_id}, Nom: {exp.name}")
    except Exception as e:
        print("Échec de la connexion au serveur MLflow:", e)

    # Start an MLflow run within the specified experiment
    with mlflow.start_run(run_name="collect_params_logging"):
        # Log the DataFrame as an artifact (CSV format)
        df_path = "res_collect_params.csv"
        df.to_csv(df_path, index=False)
        
        # Log the artifact to MLflow
        mlflow.log_artifact(df_path, artifact_path="data_frames")
        
        print(f"DataFrame {df_path} logged successfully.")

# Assuming the res_collect_params is the DataFrame you want to log
res_collect_params = collect_mlflow_model_perfs()
log_dataframe_to_mlflow(res_collect_params)
