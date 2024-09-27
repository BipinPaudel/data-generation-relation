from src.configs.config import Experiment, PredefinedRelations
from data.relation import read_relation_data, write_json_lists_to_file
from src.utils.prompt_generation import (rephrase_generation, 
                                        new_sentence_generation,
                                        generate_pseudo_label,
                                        new_sentence_generation_rem)
                                        
from src.models.model_factory import get_model
from tqdm import tqdm
from data.relation import read_json
from src.utils.utils import sanity_check

def run_experiment(cfg, experiment):
    if experiment == Experiment.REPHRASE.value:
        run_rephrase_experiment(cfg)
        
    elif experiment == Experiment.NEW_GENERATION.value:
        run_new_generation_experiment(cfg)
        
    elif experiment == Experiment.PSEUDO_LABEL_GENERATION.value:
        run_pseudo_label_experiment(cfg)
        
    elif experiment == Experiment.FIX_NEW_GEN_REMAINING.value: # Temporary remove
        fix_new_gen_remaining(cfg)
        

def run_rephrase_experiment(cfg):
    dataset_list = read_relation_data()
    prompts = []
    for data in dataset_list:
        prompt = rephrase_generation(data['text'], data['relation_name'])
        prompts.append(prompt)
    max_workers = 4
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in results:
        print('----------********-------')
        print(res[1])
        print('----------********-------\n\n\n')
        results_temp.append(res[1])
        for i, r in enumerate(results_temp):
            data = dataset_list[i]
            data['rephrase_response'] = r
            dataset_list[i]  = data
        filename = "data/results/rephrased_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)

def run_new_generation_experiment(cfg):
    dataset_list = read_relation_data()
    prompts = []
    for data in dataset_list:
        prompt = new_sentence_generation(data['text'], data['relation_name'])
        prompts.append(prompt)
    max_workers = 4
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in tqdm(results):
        print('----------********-------')
        print(res[1])
        print('----------********-------\n\n\n')
        results_temp.append(res[1])
    
        for i, r in enumerate(results_temp):
            data = dataset_list[i]
            data['new_generation_response'] = r
            dataset_list[i]  = data
        filename = "data/results/new_generation_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)
        
def run_pseudo_label_experiment(cfg):
    dataset_list = read_relation_data(filename='data/test.csv')
    prompts = []
    for data in dataset_list:
        prompt = generate_pseudo_label(data['text'],  PredefinedRelations.RELATIONS)
        prompts.append(prompt)
    max_workers = 4
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in tqdm(results):
        print('----------********-------')
        print(res[1])
        print('----------********-------\n\n\n')
        results_temp.append(res[1])
    
        for i, r in enumerate(results_temp):
            data = dataset_list[i]
            data['pseudo_label_prediction'] = r
            dataset_list[i]  = data
        filename = "data/results/psuedo_label_prediction_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)
    
    
def fix_new_gen_remaining(cfg):
    new_text_list = read_json('data/results/new_generation_relation.jsonl')
    print(len(new_text_list))

    key = 'new_generation_response'
    error_ids = sanity_check(new_text_list, key=key)
    if len(error_ids) > 0:
        print(f'problem parsing: {len(error_ids)}')
    else:
        print('all good')
    print(f'Errors: {len(error_ids)}')
    
    error_ids = [a[0] for a in error_ids]
    dataset_list = read_relation_data()
    dataset_list = [d for d in dataset_list if d['id'] in error_ids and d['relation_name'] != 'Other']
    
    prompts = []
    for data in dataset_list:
        prompt = new_sentence_generation_rem(data['text'], data['relation_name'])
        prompts.append(prompt)
    max_workers = 4
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=max_workers)
    results_temp = []
    for res in tqdm(results):
        print('----------********-------')
        print(res[1])
        print('----------********-------\n\n\n')
        results_temp.append(res[1])
    
        for i, r in enumerate(results_temp):
            data = dataset_list[i]
            data['pseudo_label_prediction'] = r
            dataset_list[i]  = data
        filename = "data/results/new_generation_relation_remaining1.jsonl"
        write_json_lists_to_file(filename, dataset_list)
        
    