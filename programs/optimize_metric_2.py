#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import time
import argparse
import os
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score, roc_auc_score, 
    accuracy_score, log_loss, matthews_corrcoef, classification_report, roc_curve
)
from sklearn.model_selection import (
    cross_validate, RepeatedKFold, RandomizedSearchCV, train_test_split
)
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from scipy.stats import spearmanr, uniform, randint, loguniform
import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import joblib
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import fbeta_score


# Add argument parsing to accept file path
def parse_args():
    parser = argparse.ArgumentParser(description="Run optimization on a dataset.")
    parser.add_argument('pathfile', type=str, help='Path to the input CSV or Excel file.')
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.pathfile
    import os
    import pandas as pd
    
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    # Start your existing workflow after loading the file
    print(f"Loading data from {input_file}...")
    if input_file.endswith('.csv'):
        X = pd.read_csv(input_file, index_col=0)
    elif input_file.endswith(('.xls', '.xlsx')):
        X = pd.read_excel(input_file)
    else:
        print("Error: Unsupported file format. Please provide a CSV or Excel file.")
        return

    # Import necessary libraries
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import os
    import shutil
    import time
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        make_scorer, precision_score, recall_score, f1_score, roc_auc_score,
        accuracy_score, log_loss, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
    )
    from sklearn.model_selection import (
        cross_validate, RepeatedKFold, RandomizedSearchCV, train_test_split
    )
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import KNNImputer
    from scipy.stats import spearmanr, uniform, randint, loguniform
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models.signature import infer_signature
    import joblib
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.calibration import calibration_curve
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import fbeta_score

    # Define functions for feature processing and model evaluation
    def feature_prop(df, add=False, formerdict=None):
        column_characteristics = formerdict if add and formerdict else {}
        for column in df.columns:
            col_data = df[column]
            if pd.api.types.is_numeric_dtype(col_data):
                col_info = {
                    'type': 'numeric',
                    'between_0_and_1': col_data.between(0, 1).all(),
                    'binary': col_data.isin([0, 1]).all(),
                    'strictly_positive': (col_data > 0).all(),
                    'strictly_negative': (col_data < 0).all()
                }
            else:
                col_info = {
                    'type': 'categorical',
                    'classes': col_data.unique().tolist()
                }
            column_characteristics[column] = col_info
        return column_characteristics

    def load_custom_variables():
        try:
            custom_vars = pd.read_csv('custom_variables.csv', sep=';')
            return custom_vars.columns.tolist()
        except FileNotFoundError:
            print("Aucun fichier custom_variables.csv détecté. Utilisation de toutes les variables.")
            return all_columns

    def which_binvariables(props):
        binaryvariables = []
        for key, values in props.items():
            if values['binary']:
                binaryvariables.append(key)
        return binaryvariables

    def draw_roc_curv_roc_auc(ytest, ypredprobroc, model_name):
        fpr, tpr, _ = roc_curve(ytest, ypredprobroc)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(ytest, ypredprobroc):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (ROC AUC)')
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_{model_name}_ROC_AUC.png")
        plt.show()

    def draw_roc_curv_fbeta(ytest, ypredprobfbeta, model_name):
        fpr, tpr, _ = roc_curve(ytest, ypredprobfbeta)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(ytest, ypredprobfbeta):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (F10)')
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_{model_name}_F10.png")
        plt.show()

    def plot_confusion_matrix_fbeta(ytest, ypredprobfbeta, model_name):
        cm_roc = confusion_matrix(ytest, ypredprobfbeta)
        disp_roc = ConfusionMatrixDisplay(confusion_matrix=cm_roc)
        disp_roc.plot()
        plt.title(f'Table de Confusion - {model_name} (F10)')
        plt.savefig(f"confusion_matrix_{model_name}_F10.png")
        plt.show()

    def plot_calibration_curve(y_true, y_prob, model_name):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(10, 7))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.title(f'Calibration Curve for {model_name}')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"calibration_curve_{model_name}.png")
        plt.show()
        mlflow.log_artifact(f"calibration_curve_{model_name}.png")

    # Start timer
    start_time = time.time()

    # Define model path
    model_path = "mlflow_model"
    if os.path.exists(model_path) and os.listdir(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    # Connect to MLFlow server
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient("http://127.0.0.1:5000")
    try:
        experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        print("Connexion au serveur MLflow. Expériences détectées:")
        for exp in experiments:
            print(f"Expérience ID: {exp.experiment_id}, Nom: {exp.name}")
    except Exception as e:
        print("Échec de la connexion au serveur MLflow:", e)

    # Prepare data
    col_caracteristics = feature_prop(df=X)
    binaryvariables = which_binvariables(col_caracteristics)
    OHEprefix = binaryvariables
    data_train = X
    data_train_sample = data_train.sample(frac=0.25)
    sample_predictors = data_train_sample.drop(["TARGET"], axis=1)
    sample_target = data_train_sample["TARGET"]
    all_columns = sample_predictors.columns
    numeric_columns = [col for col in all_columns if not any(col.startswith(prefix) for prefix in OHEprefix)]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns)
        ],
        remainder='passthrough'
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(sample_predictors, sample_target, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    signature = infer_signature(X_train, y_train)

    # Define models
    models = [
        ("LogisticRegression", LogisticRegression(solver='liblinear', max_iter=500), {
            'model__C': uniform(0.01, 2),
            'model__penalty': ['l1', 'l2']
        })
    ]

    performance_df = pd.DataFrame(columns=[
        'Model', 'Train Accuracy', 'Train F1 Score', 'Train ROC AUC', 'Train F10 Score',
        'Test Accuracy', 'Test F1 Score', 'Test ROC AUC', 'Test F10 Score',
        'Best Params', 'CV Train Mean Score', 'CV Train Std Dev', 'CV Test Mean Score', 'CV Test Std Dev'
    ])
    start_time = time.time()

    for model_name, model, param_dist in models:
        print(f"Entraînement pour le modèle: {model_name}")
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])
        f10_scorer = make_scorer(fbeta_score, beta=10)

        print(f"Grid Search CV pour {model_name} avec F10")
        with mlflow.start_run(run_name=f"{model_name}_F10"):
            random_search_fbeta = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=10,
                scoring=f10_scorer,
                cv=5,
                verbose=2,
                random_state=42,
                n_jobs=-1,
                return_train_score=True
            )

            random_search_fbeta.fit(X_train, y_train)
            best_params_fbeta = random_search_fbeta.best_params_
            print("Best params fbeta:", best_params_fbeta)

            y_pred_train_fbeta = random_search_fbeta.best_estimator_.predict(X_train)
            y_pred_prob_train_fbeta = random_search_fbeta.best_estimator_.predict_proba(X_train)[:, 1]
            y_pred_test_fbeta = random_search_fbeta.best_estimator_.predict(X_test)
            y_pred_prob_test_fbeta = random_search_fbeta.best_estimator_.predict_proba(X_test)[:, 1]

            train_fbeta = fbeta_score(y_train, y_pred_train_fbeta, beta=10)
            test_fbeta = fbeta_score(y_test, y_pred_test_fbeta, beta=10)

            mlflow.log_params(best_params_fbeta)
            mlflow.log_metrics({
                "train_fbeta_score": train_fbeta,
                "test_fbeta_score": test_fbeta
            })

            mlflow.sklearn.log_model(random_search_fbeta.best_estimator_, f"{model_name}_F10_model")

            draw_roc_curv_fbeta(y_test, y_pred_prob_test_fbeta, model_name)
            mlflow.log_artifact(f"roc_curve_{model_name}_F10.png")
            plot_confusion_matrix_fbeta(y_test, y_pred_test_fbeta, model_name)
            mlflow.log_artifact(f"confusion_matrix_{model_name}_F10.png")

            plot_calibration_curve(y_test, y_pred_prob_test_fbeta, model_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps d'exécution total: {elapsed_time:.2f} secondes")
    mlflow.end_run()

if __name__ == "__main__":
    main()

