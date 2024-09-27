from src.configs import ModelConfig
from .model import BaseModel
from .hf_model import HFModel
from .open_ai import OpenAIGPT

def get_model(config: ModelConfig) -> BaseModel:
    print(f'model factory: {config.provider}')
    if config.provider == "hf":
        return HFModel(config)
    elif config.provider == "openai":
        return OpenAIGPT(config)
    else:
        raise NotImplementedError
    