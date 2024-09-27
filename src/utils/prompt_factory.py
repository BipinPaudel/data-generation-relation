from .water_prompt_generation import WaterPrompt
from .semeval_prompt_generation import SemEvalPrompt


def get_prompt_obj(dataset):
    print(f'dataset factory: {dataset}')
    if dataset == "water":
        return WaterPrompt()
    elif dataset == "semeval":
        return SemEvalPrompt()
    else:
        raise NotImplementedError
    