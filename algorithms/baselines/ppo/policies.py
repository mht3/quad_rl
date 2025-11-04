import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM feature extractor for use with LSTM policies.
    
    :param observation_space: The observation space
    :param features_dim: Number of features to extract
    :param lstm_hidden_size: Hidden size of the LSTM
    :param n_lstm_layers: Number of LSTM layers
    :param activation_fn: Activation function
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 64,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(observation_space, features_dim)
        
        # Flatten the observation space
        input_dim = observation_space.shape[0] if isinstance(observation_space, spaces.Box) else observation_space.n
        
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True
        )
        
        # Output layer to get desired features_dim
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            activation_fn()
        )
        
        # Initialize hidden states
        self.hidden_state = None
        self.cell_state = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM extractor.
        
        :param observations: Input observations
        :return: Extracted features
        """
        batch_size = observations.shape[0]
        
        # If this is the first call or batch size changed, initialize hidden states
        if (self.hidden_state is None or 
            self.hidden_state.shape[1] != batch_size):
            self.reset_states(batch_size, observations.device)
        
        # Ensure observations have sequence dimension (batch_size, seq_len=1, features)
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        
        # Forward through LSTM
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            observations, (self.hidden_state, self.cell_state)
        )
        
        # Take the last output (since seq_len=1, this is just squeezing)
        lstm_out = lstm_out.squeeze(1)
        
        # Forward through output layer
        features = self.output_layer(lstm_out)
        
        return features
    
    def reset_states(self, batch_size: int, device: torch.device):
        """Reset the LSTM hidden and cell states."""
        self.hidden_state = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=torch.float32
        )
        self.cell_state = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=torch.float32
        )


class LSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with LSTM feature extraction.
    
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture for actor and critic
    :param activation_fn: Activation function
    :param lstm_hidden_size: Hidden size of the LSTM
    :param n_lstm_layers: Number of LSTM layers
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        *args,
        **kwargs,
    ):
        # Store LSTM parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        
        # Create LSTM feature extractor
        features_extractor_class = LSTMExtractor
        features_extractor_kwargs = {
            "features_dim": lstm_hidden_size,
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
            "activation_fn": activation_fn,
        }
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )
    
    def reset_lstm_states(self, batch_size: int = 1):
        """Reset LSTM states. Call this at the beginning of each episode."""
        if hasattr(self.features_extractor, 'reset_states'):
            device = next(self.parameters()).device
            self.features_extractor.reset_states(batch_size, device)
    
    def predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        state: Optional[torch.Tensor] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """
        Override predict to handle LSTM state resets.
        
        :param observation: Input observation
        :param state: Not used (kept for compatibility)
        :param episode_start: Whether this is the start of an episode
        :param deterministic: Whether to use deterministic actions
        :return: Predicted action
        """
        # Reset LSTM states if this is the start of an episode
        if episode_start is not None and episode_start.any():
            batch_size = observation.shape[0] if isinstance(observation, torch.Tensor) else 1
            self.reset_lstm_states(batch_size)
        
        return super().predict(observation, state, episode_start, deterministic)