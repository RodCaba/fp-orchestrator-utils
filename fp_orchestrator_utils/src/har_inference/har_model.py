import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class HARModel(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128, num_layers: int = 2):
        super(HARModel, self).__init__()
        self.num_classes = num_classes
        self.imu_feature_size = 64  # All IMU sensors output 64-dimensional features

        # Sepearate processors for each IMU sensor type
        self.accelerometer_processor = IMUSensorProcessor(3, self.imu_feature_size, hidden_size, num_layers)
        self.gyroscope_processor = IMUSensorProcessor(3, self.imu_feature_size, hidden_size, num_layers)
        self.total_acceleration_processor = IMUSensorProcessor(3, self.imu_feature_size, hidden_size, num_layers)
        self.gravity_processor = IMUSensorProcessor(3, self.imu_feature_size, hidden_size, num_layers)
        self.orientation_processor = IMUSensorProcessor(7, self.imu_feature_size, hidden_size, num_layers)

        # Audio processor
        self.audio_processor = AudioProcessor(output_size=self.imu_feature_size, hidden_size=hidden_size, num_layers=num_layers)

        # Attention mechanism to fuse multi-modal features
        self.feature_attention = MultiModalAttention(input_size=self.imu_feature_size, num_modalities=6)

        total_feature_size = self.imu_feature_size * 6 + 1  # 6 sensors + n_users
        self.classifier = nn.Sequential(
            nn.Linear(total_feature_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(
            self,
            sensor_data: dict[str, torch.Tensor],
            n_users: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with variable-length sequences.
        """
        features = []

        if 'accelerometer' in sensor_data:
            acc_feat = self.accelerometer_processor(sensor_data['accelerometer'])
            features.append(acc_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        if 'gyroscope' in sensor_data:
            gyro_feat = self.gyroscope_processor(sensor_data['gyroscope'])
            features.append(gyro_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        if 'total_acceleration' in sensor_data:
            total_acc_feat = self.total_acceleration_processor(sensor_data['total_acceleration'])
            features.append(total_acc_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        if 'gravity' in sensor_data:
            gravity_feat = self.gravity_processor(sensor_data['gravity'])
            features.append(gravity_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        if 'orientation' in sensor_data:
            orient_feat = self.orientation_processor(sensor_data['orientation'])
            features.append(orient_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        # Process audio if available
        if 'audio' in sensor_data:
            audio_feat = self.audio_processor(sensor_data['audio'])
            features.append(audio_feat)
        else:
            features.append(torch.zeros(sensor_data[next(iter(sensor_data))].size(0), self.imu_feature_size))

        # Apply attention to fuse features
        attended_features = self.feature_attention(features)
        # Add n_users feature
        n_users_expanded = n_users.float().unsqueeze(1)  # Ensure shape (batch_size, 1)

        # Concatenate all features
        combined_features = torch.cat(attended_features + [n_users_expanded], dim=1)
        output = self.classifier(combined_features)

        return output

class IMUSensorProcessor(nn.Module):
    """
    Process individual IMU sensor with variable-length sequences.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ):
        super(IMUSensorProcessor, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
           hidden_size,
          hidden_size // 2,
          num_layers,
          batch_first=True,
          bidirectional=True,
          dropout= 0.2 if num_layers > 1 else 0.0
        )
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for IMU sensor data.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        :return: Processed tensor of shape (batch_size, output_size)
        """
        x = self.input_projection(x)
        x, _ = self.lstm(x)
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)
        x = self.output_projection(self.dropout(x))

        return x
    
class AudioProcessor(nn.Module):
    """
    Process audio features.
    """
    def __init__(self,
                 output_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2
                 ):
        super(AudioProcessor, self).__init__()

        self.input_size = 126  # Assuming 126 mel bands
        self.input_projection = nn.Linear(self.input_size, hidden_size)
        self.lstm = nn.LSTM(
          hidden_size,
          hidden_size // 2,
          num_layers,
          batch_first=True,
          bidirectional=True,
          dropout= 0.2 if num_layers > 1 else 0.0
        )
        self.temporal_lstm = nn.LSTM(
          hidden_size,
          hidden_size // 2,
          num_layers,
          batch_first=True,
          bidirectional=True,
          dropout= 0.2 if num_layers > 1 else 0.0
        )

        self.output_projection = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio data.
        :param x: Input tensor of shape (batch_size, time_steps, n_mels)
        :return: Processed tensor of shape (batch_size, output_size)
        """
        batch_size, num_segments, time_steps, n_mels = x.shape
        # Reshape to process segments together
        x = x.view(batch_size * num_segments, time_steps, n_mels)
      
        x = self.input_projection(x)
        x, _ = self.lstm(x)

        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, num_segments, -1)  # Reshape back
        # Process across segments
        x, _ = self.temporal_lstm(x)

        x = torch.mean(x, dim=1)
        x = self.output_projection(self.dropout(x))

        return x
    
class MultiModalAttention(nn.Module):
    """
    Attention mechanism to fuse multi-modal features.
    """
    def __init__(self, input_size: int, num_modalities: int):
        super(MultiModalAttention, self).__init__()
        
        self.attention_weights = nn.Linear(input_size, 1)
        self.num_modalities = num_modalities

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass for attention mechanism.
        :param features: List of feature tensors from different modalities.
        :return: Fused feature tensor.
        """
        stacked_features = torch.stack(features, dim=1)

        # Compute attention scores
        attention_scores = self.attention_weights(stacked_features)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention
        attended = stacked_features * attention_weights

        # Back to list
        attended_features = [attended[:, i, :] for i in range(self.num_modalities)]
        return attended_features