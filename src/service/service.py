from src.configs.config import Experiment, PredefinedRelations
from data.relation import read_relation_data, write_json_lists_to_file, read_relation_data_from_final_file
from src.utils.prompt_factory import get_prompt_obj
                                        
from src.models.model_factory import get_model
from tqdm import tqdm
from data.relation import read_json
from src.utils.utils import sanity_check
from src.utils.helper_model import SimcseModel

def run_experiment(cfg, experiment, dataset, data_type):
    if experiment == Experiment.REPHRASE.value:
        run_rephrase_experiment(cfg, dataset, data_type)
        
    elif experiment == Experiment.NEW_GENERATION.value:
        run_new_generation_experiment(cfg, dataset, data_type)
        
    elif experiment == Experiment.PSEUDO_LABEL_GENERATION.value:
        run_pseudo_label_experiment(cfg, dataset, data_type)
        
    elif experiment == Experiment.FIX_NEW_GEN_REMAINING.value: # Temporary remove
        fix_new_gen_remaining(cfg, dataset)

    elif experiment == Experiment.FEW_SHOT_PSEUDO_LABEL_GENERATION.value:
        run_few_shot_pseudo_label_experiment(cfg, dataset, data_type)
        

def run_rephrase_experiment(cfg, dataset, data_type):
    dataset_list = read_relation_data(dataset=dataset, data_type=data_type)
    prompt_obj = get_prompt_obj(dataset)
    prompts = []
    for data in dataset_list:
        prompt = prompt_obj.rephrase_generation(data['text'], data['relation_name'])
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
        filename = f"data/results/{dataset}/{dataset}_rephrased_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)

def run_new_generation_experiment(cfg, dataset, data_type):
    dataset_list, id_relation_dict = read_relation_data(dataset=dataset, data_type=data_type)
    prompt_obj = get_prompt_obj(dataset)
    prompts = []
    for data in dataset_list:
        prompt = prompt_obj.new_sentence_generation(data['text'], data['relation_name'])
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

        filename = f"data/results/{dataset}/{dataset}_new_generation_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)
        
def run_pseudo_label_experiment(cfg, dataset, datatype):
    dataset_list, id_relation_dict = read_relation_data(dataset=dataset, data_type=datatype)
    prompts = []
    predefined_relations = list(id_relation_dict.values())
    prompt_obj = get_prompt_obj(dataset)
    for data in dataset_list:
        prompt = prompt_obj.generate_pseudo_label(data['text'],  predefined_relations)
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
        filename = f"data/results/{dataset}_{datatype}_psuedo_label_prediction_relation.jsonl"
        write_json_lists_to_file(filename, dataset_list)
    
    
def fix_new_gen_remaining(cfg, dataset):
    new_text_list = read_json('data/results/new_generation_relation.jsonl')
    print(len(new_text_list))
    prompt_obj = get_prompt_obj(dataset)
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
        prompt = prompt_obj.new_sentence_generation_rem(data['text'], data['relation_name'])
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
        
def run_few_shot_pseudo_label_experiment(cfg, dataset, data_type):
    
    
    dataset_list, id_relation_dict = read_relation_data_from_final_file(dataset=dataset, datatype=data_type)
    sent_model = SimcseModel(id_relation_dict, dataset=dataset)
    

    predefined_relations = list(id_relation_dict.values())
    prompt_obj = get_prompt_obj(dataset)
    prompts = []
    for ds in dataset_list:
        ex = sent_model.get_sim_examples(ds['text']) #This comes from sim eval prompt

        prompt = prompt_obj.generate_few_shot_pseudo_label(ds['text'], predefined_relations, id_relation_dict, ex)

        prompts.append(prompt)

    
    model_name = cfg.gen_model.name.split('/')[-1]
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
            data['few_shot_pseudo_label_prediction'] = r
            dataset_list[i]  = data
        filename = f"data/results/{dataset}_{data_type}_15shot_label_{model_name}.json"
        write_json_lists_to_file(filename, dataset_list)
