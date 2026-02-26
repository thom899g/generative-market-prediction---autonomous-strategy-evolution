"""
Configuration for the Generative Market Prediction System
Centralized configuration with environment variable support
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GANConfig:
    """GAN Model Configuration"""
    latent_dim: int = 100
    generator_hidden: List[int] = [256, 512, 256]
    discriminator_hidden: List[int] = [256, 128, 64]
    learning_rate: float = 0.0002
    batch_size: int = 64
    epochs: int = 5000
    sequence_length: int = 60  # Lookback window
    features: int = 5  # OHLC + volume
    dropout_rate: float = 0.3
    
@dataclass
class TradingConfig:
    """Trading Strategy Configuration"""
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    risk_free_rate: float = 0.02
    stop_loss_pct: float =