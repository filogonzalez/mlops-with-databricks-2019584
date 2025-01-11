import mlflow

client = mlflow.tracking.MlflowClient()
schema = "hotel_cancellation"

src_model_name = "stg.{schema}.model"
src_model_version = "1"
src_model_uri = f"models:/{src_model_name}/{src_model_version}"
dst_model_name = "prd.{schema}.model"

copied_model_version = client.copy_model(src_model_uri, dst_model_name)