# Databricks notebook source
import mlflow
import numpy as np
import yaml
from pyspark.sql import SparkSession
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")


# COMMAND ----------
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

num_features = config.get("num_features")
cat_features = config.get("cat_features")
target = config.get("target")
parameters = config.get("parameters")
catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/hotel-cancellations-basic"],
    filter_string="tags.branch='02_04'",
).run_id[0]
model = mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------
model.predict(X=X_test[0:1])

# COMMAND ----------

class HotelCancellationWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        self.classes = ['Cancelled', 'Not cancelled']
        
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        return {"Prediction": np.array([self.classes[x] for x in predictions])[0]}
    

# COMMAND ----------
wrapped_model = HotelCancellationWrapper(model)
wrapped_model.predict(context=None, model_input=X_test[0:1])


# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotel-cancellations-basic")

with mlflow.start_run(
    tags={"branch": "02_05"},
) as run:
    
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, 
                                model_output={'Prediction': 'Cancelled'})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-model",
        signature=signature,
    )

# COMMAND ----------
