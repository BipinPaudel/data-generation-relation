from src.prompts import Prompt

class TacredPrompt:
    
    def new_sentence_generation(self, original_text, relation_name) -> Prompt:
        header = "Please generate an entire new sentence with a different context while preserving the specified relationship. "\
                f"Here's a sentence:\n\n {original_text}\n"
        header += f"The words inside the tag e1 and e2 are the two entities present in the text and their relationship is:\n{relation_name}\n"

        # print(original_text, relation_name)
        # if relation_name == 'Other':
        #     header += ''
        # elif relation_name.index('e1') < relation_name.index('e2'):
        #     header += "In this relationship, the two words separated by - denotes the relationship direction from entity e1 to e2 in the given sentence"
        # else:
        header += "The relationship denotes the relation direction from entity e1 to e2 in the given sentence"\
        
        # if relation_name != 'Other':
        #     footer = "Please output the new sentence by keeping the entities and their relationship intact, meaning the word in entities should be exactly same "\
        #         f"as given: {relation_name}. The direction flow of relation should be maintained in the output and the new sentence should be inside [CLS] [SEP] as given above."
        # else:
        footer = "Please output the new sentence by keeping the entities intact, meaning the word in entities should be exactly same inside the tag. "\
                "Also, the new sentence should be inside [CLS] [SEP] as given above."
            
        system_prompt = "You are an expert in English language task in generating a sentence based "\
                "on the entities and their relationship."
            
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate="",
            footer=footer,
            original_point=None,  # type: ignore
            gt=None,  # type: ignore
            answer="",
            shots=[],
            id=None,  # type: ignore
        )
        return prompt
    
    def generate_pseudo_label(self, original_text, predefined_relations) -> Prompt:
        header = f"Analyze the given sentence carefully and determine how the entities inside the tags e1 and e2 are related. "\
        
        
        footer = "Please output the relationship between entities in the format below:\n <relation> </relation>\n "\
                f"Select only one option from the predefined relations list given in the list below: \n{predefined_relations}\n" \
                    "The relationship in the list denote the flow of relation from entity e1 to e2 or e2 to e1. "\
                "Also, output the confidence score (0-1) indicating your certainty about the relationship inside the <confidence> tag."
        
        footer += f"\nTest Sentence:\n{original_text}\n is"
        
        system_prompt = "You are an expert in natural language processing, specializing in understanding relationships between entities in text. Your task is to identify and output the relationship between two entities enclosed within the tags <e1> and <e2>."
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate="",
            footer=footer,
            original_point=None,  # type: ignore
            gt=None,  # type: ignore
            answer="",
            shots=[],
            id=None,  # type: ignore
        )
        return prompt

    
    def generate_few_shot_pseudo_label(self, original_text, predefined_relations, id_relation_dict, ex) -> Prompt:
        header = f"Analyze the given test sentence carefully and determine how the entities inside the "\
            f"tags e1 and e2 are related."

        system_prompt = f'''You are an expert in natural language processing, specializing in understanding 
        relationships between entities in text. Your task is to identify and output the relationship 
        between two entities enclosed within the tags <e1> and <e2>'''

        footer = "Please output the relationship between entities in the format below:\n <relation> </relation>\n "\
        f"Select only one option from the predefined relations list given in the list below: \n{predefined_relations}\n " \
        "The relationship in the list denote the flow of relation from entity e1 to e2 or e2 to e1. "\
        "Also, output the confidence score (0-1) indicating your certainty about the relationship inside the <confidence> tag."\
        "Use the following demonstrations as examples:"

        if ex != None or ex != []:
            for example in ex:
                rel = id_relation_dict.get(example[1])

                footer += f'''
                Example Sentence: {example[0]} is 
                <relation>{rel}</relation>\n<confidence>0.8</confidence>
                '''
        
        footer += f"\nTest Sentence:\n{original_text}\n is"
        
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate="",
            footer=footer,
            original_point=None,  # type: ignore
            gt=None,  # type: ignore
            answer="",
            shots=[],
            id=None,  # type: ignore
        )
        return prompt