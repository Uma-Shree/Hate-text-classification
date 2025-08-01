import sys
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.gcloud = GCloudSync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Method to initiate model pusher
        """
        try:
            # uploading th model to gcloud storage
            self.gcloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,
                                              self.model_pusher_config.TRAINED_MODEL_PATH,
                                              self.model_pusher_config.MODEL_NAME)
            
            model_pusher_artifacts = ModelPusherArtifacts(
                bucket_name = self.model_pusher_config.BUCKET_NAME
                
            )
            #logging.info(f"Model pusher artifacts: {model_pusher_artifacts}")
            #logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifacts
        
        except Exception as e:
            raise CustomException(e, sys) from e