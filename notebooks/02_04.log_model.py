# Databricks notebook source
import mlflow
import yaml
from pyspark.sql import SparkSession
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------
with open("/Users/filo.gzz/Desktop/Filo/Projects/mastercard/mlops/LinkedinMLOps/mlops-with-databricks/mlops-with-databricks-2019584/project_config.yml", "r") as file:
    config = yaml.safe_load(file)

num_features = config.get("num_features")
cat_features = config.get("cat_features")
target = config.get("target")
parameters = config.get("parameters")
catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), cat_features)], remainder="passthrough"
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))]
)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotel-cancellations-basic")

with mlflow.start_run(
    tags={"git_sha": "51v63531711eaa139",
          "branch": "02_04"},
    description="demo run",
) as run:
    
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    print(f"Classification Report: \n{clf_report}")

    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm-pipeline-model",
        signature=signature,
    )
# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
