from enum import Enum
from pydantic import BaseModel as PBM
from pydantic import Field
from typing import Any, Dict, List, Optional

class Experiment(Enum):
    REPHRASE = 'rephrase'
    NEW_GENERATION = 'new_generation'
    PSEUDO_LABEL_GENERATION = 'psuedo_label_generation'
    FIX_NEW_GEN_REMAINING = 'fix_new_gen_remaining'

class PredefinedRelations:
    RELATIONS = [
    "Component-Whole(e2, e1)",
    "Component-Whole(e1, e2)",
    "Instrument-Agency(e2, e1)",
    "Instrument-Agency(e1, e2)",
    "Member-Collection(e1, e2)",
    "Member-Collection(e2, e1)",
    "Cause-Effect(e2, e1)",
    "Cause-Effect(e1, e2)",
    "Entity-Destination(e1, e2)",
    "Entity-Destination(e2, e1)",
    "Content-Container(e1, e2)",
    "Content-Container(e2, e1)",
    "Message-Topic(e1, e2)",
    "Message-Topic(e2, e1)",
    "Product-Producer(e2, e1)",
    "Product-Producer(e1, e2)",
    "Entity-Origin(e1, e2)",
    "Entity-Origin(e2, e1)",
    "Other"
]

class Task(Enum):
    ACS = "ACS"  # American Community Survey
    PAN = "PAN"  # PAN 2018
    REDDIT = "REDDIT"  # Reddit
    # Synthetic options
    CHAT = "CHAT"  # Multiturn conversations
    CHAT_EVAL = "CHAT_EVAL"  # Evaluation of multiturn conversations
    SYNTHETIC = "SYNTHETIC"  # Synthetic Reddit comments generation

class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field(description="Provider of the model")
    dtype: str = Field(
        "float16", description="Data type of the model (only used for local models)"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)",
    )
    prompt_template: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the prompt"
    )
    submodels: List["ModelConfig"] = Field(
        default_factory=list, description="Submodels to use"
    )
    multi_selector: str = Field(
        default="majority", description="How to select the final answer"
    )

    def get_name(self) -> str:
        if self.name == "multi":
            return "multi" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.name == "chain":
            return "chain_" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.provider == "hf":
            return self.name.split("/")[-1]
        else:
            return self.name
        
        
class SYNTHETICConfig(PBM):
    path: str = Field(default=None, description='location of synthetic dataset')
    
    individual_prompts: bool = Field(
        default=False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )
    
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    
    num_of_guesses: int = Field(
        default=1,
        description='Number of guesses by the model'
    )
    
    reasoning: str = Field(
        default=False,
        description='Reasoning in the output of the model'
    )

class Config(PBM):
    task_config: SYNTHETICConfig = Field(default=None, description="Config for the task")
    
    gen_model: ModelConfig = Field(
        default=None, description="Model to use for generation, ignored for CHAT task"
    )
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    
    