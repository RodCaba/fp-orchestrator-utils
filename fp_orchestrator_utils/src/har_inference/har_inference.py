import logging
import os
import torch
import numpy as np
from fp_orchestrator_utils.storage.s3 import S3Service, S3Config
from fp_orchestrator_utils.src.data_loader import DataLoader
import onnxruntime as ort

logger = logging.getLogger(__name__)

class HARInference:
   """
   HAR Inference Engine.

   Loads a trained HAR model and performs inference on new data.
   """
   
   # Activity class mapping based on the training data
   CLASS_NAMES = {
       0: "Cleaning",
       1: "Cooking", 
       2: "Eating",
       3: "Eating - Watching TV",
       4: "Playing",
       5: "Watching TV"
   }

   def __init__(self, device: str = 'cpu', load_from_s3: bool = True):
      s3_config = S3Config(
          access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
          secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
          bucket_name=os.getenv("S3_BUCKET_NAME", ""),
      )

      self.s3_service = S3Service(s3_config)
      self.model_prefix = os.getenv("S3_MODEL_PREFIX", "har_model/")
      self.device = device
      # Load ONNX model from S3
      self.session = None
      self.load_model(load_from_s3)

      if self.session is None:
          raise RuntimeError("Failed to load the HAR model for inference.")
      

    
   def load_model(self, load_from_s3: bool = True, local_path: str = "har_model.onnx") -> bool:
        """
        Loads the ONNX model from S3 or local path.
        :param load_from_s3: Flag to indicate whether to load the model from S3.
        :param local_path: Local path to load the model from if not loading from S3.
        :return: True if the model was loaded successfully, False otherwise.
        """
        try:
            if load_from_s3:
                s3_key = os.path.join(self.model_prefix, "har_model.onnx")
                model_data = self.s3_service.load(s3_key)
                with open(local_path, 'wb') as f:
                    f.write(model_data)
                logger.info(f"Model downloaded from S3 and saved to {local_path}")
            else:
                if not os.path.exists(local_path):
                    logger.error(f"Local model file {local_path} does not exist.")
                    return False
                logger.info(f"Loading model from local path {local_path}")

            self.session = ort.InferenceSession(local_path, providers=['CPUExecutionProvider'])
            logger.info("ONNX model loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
       
        
   def _collate_for_inference(self, batch: list[dict]) -> tuple:
       """
       Custom collate function for inference with consistent padding for ONNX.
       Based on the existing collate_variable_length but ensures all sequences have the same length.
       """
       sensor_data = {}
       n_users_list = []
       labels_list = []

       # Get all sensor types from the first sample
       sample_sensors = batch[0]['features'].keys()
       
       # Find the maximum sequence length across all sensors in the batch
       max_seq_length = 0
       for sample in batch:
           for sensor_type in sample_sensors:
               if sensor_type != 'audio' and hasattr(sample['features'][sensor_type], 'shape'):
                   max_seq_length = max(max_seq_length, sample['features'][sensor_type].shape[0])
       
       # Ensure minimum sequence length for model compatibility  
       max_seq_length = max(max_seq_length, 50)

       for sensor_type in sample_sensors:
           if sensor_type == 'audio':
               audio_tensors = [torch.tensor(sample['features']['audio'], dtype=torch.float32) for sample in batch]
               padded_sequences = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)
               
               # Ensure exactly 5 audio segments for ONNX compatibility
               if padded_sequences.shape[1] > 5:
                   padded_sequences = padded_sequences[:, :5]  # Take first 5
               elif padded_sequences.shape[1] < 5:
                   padding_needed = 5 - padded_sequences.shape[1]
                   padding_shape = (padded_sequences.shape[0], padding_needed, padded_sequences.shape[2], padded_sequences.shape[3])
                   padding = torch.zeros(padding_shape, dtype=torch.float32)
                   padded_sequences = torch.cat([padded_sequences, padding], dim=1)
               
               sensor_data['audio'] = padded_sequences
           else:
               # Variable-length sensors - pad all to the same maximum length
               sequences = []
               for sample in batch:
                   sensor_seq = sample['features'][sensor_type]
                   # Create a zero-padded tensor of max_seq_length
                   if sensor_type == 'orientation':
                       padded_seq = np.zeros((max_seq_length, 7), dtype=np.float32)
                   else:
                       padded_seq = np.zeros((max_seq_length, 3), dtype=np.float32)
                   
                   # Copy actual data up to its length
                   actual_length = min(sensor_seq.shape[0], max_seq_length)
                   padded_seq[:actual_length] = sensor_seq[:actual_length]
                   
                   sequences.append(torch.tensor(padded_seq, dtype=torch.float32))
               
               sensor_data[sensor_type] = torch.stack(sequences)

       # Collect n_users and labels
       for item in batch:
           n_users_list.append(item['n_users'])
           labels_list.append(item['label'])

       n_users_tensor = torch.tensor(n_users_list, dtype=torch.float32)
       labels_tensor = torch.tensor(labels_list, dtype=torch.long) if labels_list else None

       return sensor_data, n_users_tensor, labels_tensor

   def predict(self, input_data: list[dict]) -> dict:
      """
      Perform inference on the input data.

      Args:
          input_data (list[dict]): List of dictionaries containing sensor data and n_users.
      Returns:
          dict: Dictionary with predictions and associated probabilities.
      """
      # Preprocess data using the DataLoader's preprocess_data method
      data_loader = DataLoader()
      _, processed_data = data_loader.process_sensor_data(input_data)
      
      # Create a batch with single sample for the collate function
      # For inference, we don't have a real label, so we'll use a dummy label
      batch = [{
          'features': processed_data['features'],
          'n_users': processed_data['n_users'],
          'label': 0  # Dummy label for inference
      }]
      
      # Use our custom collate function that ensures consistent padding for ONNX
      sensor_data, n_users_tensor, _ = self._collate_for_inference(batch)
      
      # Prepare inputs for ONNX model - convert tensors to numpy and extract individual sensors
      ort_inputs = {}
      
      # Convert each sensor tensor to numpy array for ONNX
      for sensor_type in ['accelerometer', 'gyroscope', 'totalacceleration', 'gravity', 'orientation']:
          if sensor_type in sensor_data:
              ort_inputs[sensor_type] = sensor_data[sensor_type].numpy().astype(np.float32)
          else:
              # Create zero tensor if sensor is missing
              if sensor_type == 'orientation':
                  ort_inputs[sensor_type] = np.zeros((1, 1, 7), dtype=np.float32)
              else:
                  ort_inputs[sensor_type] = np.zeros((1, 1, 3), dtype=np.float32)
      
      # Handle audio separately - the collate function handles padding
      if 'audio' in sensor_data:
          audio_tensor = sensor_data['audio'].numpy().astype(np.float32)
          # Ensure we have exactly 5 segments for the ONNX model
          if audio_tensor.shape[1] > 5:
              audio_tensor = audio_tensor[:, :5]  # Take first 5 segments
          elif audio_tensor.shape[1] < 5:
              # Pad with zeros to reach 5 segments
              padding_shape = (audio_tensor.shape[0], 5 - audio_tensor.shape[1], audio_tensor.shape[2], audio_tensor.shape[3])
              padding = np.zeros(padding_shape, dtype=np.float32)
              audio_tensor = np.concatenate([audio_tensor, padding], axis=1)
          ort_inputs['audio'] = audio_tensor
      else:
          ort_inputs['audio'] = np.zeros((1, 5, 64, 126), dtype=np.float32)
      
      # Add n_users 
      ort_inputs['n_users'] = n_users_tensor.numpy().astype(np.float32)
      
      # Run inference
      ort_outs = self.session.run(None, ort_inputs)
      predictions = ort_outs[0]
      probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
      predicted_classes = predictions.argmax(axis=1)
      
      # Convert class indices to names
      predicted_class_names = [self.CLASS_NAMES.get(class_idx, f"Unknown_{class_idx}") 
                              for class_idx in predicted_classes]
      
      return {
          'predictions': predicted_classes.tolist(),
          'predicted_class_names': predicted_class_names,
          'probabilities': probabilities.tolist(),
      }
