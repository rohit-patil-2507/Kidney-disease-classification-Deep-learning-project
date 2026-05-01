import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        import os
        # Set authentication credentials for remote DagsHub MLflow server
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'rohit-patil-2507'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'eceeb5ddbc868faee671550f4c23909cdd1b7e29'
        
        # Use remote MLflow tracking server (DagsHub)
        mlflow.set_tracking_uri("https://dagshub.com/rohit-patil-2507/Kidney-disease-classification-Deep-learning-project.mlflow")

        experiment_name = "Kidney-Disease-Classification"
        
        # Check if experiment exists and is not deleted
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is not None and exp.lifecycle_stage == "deleted":
            # Restore the deleted experiment using MlflowClient
            client = MlflowClient()
            client.restore_experiment(exp.experiment_id)
        elif exp is None:
            # Create new experiment if it doesn't exist
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Register model only if using remote tracking server
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, 
                    "model", 
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
