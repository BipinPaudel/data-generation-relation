from data.relation import read_json
from src.utils.utils import sent_emb_entity
from simcse import SimCSE

class SimcseModel:

    def __init__(self,relation_dict_whole, dataset) -> None:
        self.mod_pmpt_train = None
        self.mod_sent_train = None
        self.label_idx = None
        self.model = None
        self.dataset = dataset
        self.relation_dict_whole = relation_dict_whole
        self.generate_sim_cse_model()

    def generate_sim_cse_model(self, ):
        if self.dataset == 'semeval':
            data = read_json('/home/n/ngautam/researchscripts/MetaSRE/data/SemEval/train_sentence.json')
            data2 = read_json('/home/n/ngautam/researchscripts/MetaSRE/data/SemEval/train_label_id.json')
        elif self.dataset == 'tacred':
            data = read_json('data/tacred/train_sentence.json')
            data2 = read_json('data/tacred/train_label_id.json')
        elif self.dataset == 'water':
            data = read_json('data/water/train/train_sentence.json')
            data2 = read_json('data/water/train/train_label_id.json')
            
            
        
        mod_sent_train = []
        mod_pmpt_train = []
        sent_train = []

        label_idx = []
        for d in data2:
            label_idx.append(d)

        for index, d in enumerate(data):
            mod = d.split("[CLS]")[1].split("[SEP]")[0]
            sent_train.append(mod)
            ret_text, tex = sent_emb_entity(mod, label_idx[index], self.relation_dict_whole)
            mod_sent_train.append(ret_text)
            mod_pmpt_train.append(tex)

        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model.build_index(mod_sent_train)

        self.mod_pmpt_train = mod_pmpt_train
        self.mod_sent_train = mod_sent_train
        self.label_idx = label_idx
        return self.model
    
    def get_sim_examples(self, text):
        shot = 16
        results = self.model.search(text, threshold=0.0, top_k=shot+5)

        if results != []:
            ex = []
            try:
                res = results[1:shot]
                for r in res:
                    idx = self.mod_sent_train.index(f'{r[0]}')
                    label = self.label_idx[idx]
                    # for key, value in self.relation_dict.items():
                    #     if key == label:
                    #         our_key = value
                    #         break
                    match = self.mod_pmpt_train[idx]
                    ex.append([match, label])
                    # ex.append([r[0], our_key])
            except Exception as e:
                return ex

            return ex

        return None
