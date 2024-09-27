from src.prompts import Prompt


def rephrase_generation(original_text, relation_name) -> Prompt:
    header = f"Rephrase the following sentence to enhance its complexity while preserving the " \
            f"specified relationship. Here's a sentence:\n\n {original_text}"
    header += f"Here's a relationship:\n\n{relation_name}\n\n"
          
    header += "The relationship denotes how the two entities inside the tags e1 and e2"\
            "are related to eachother." 
        
    footer = "Please output the rephrased sentence while not chainging the word of the entities "\
            " which are given inside <e1> and <e2> tags and keeping their relationship intact. "\
                "The sentence should be inside [CLS] [SEP] as given above."
    system_prompt = "You are an expert in English language task in rephrasing the text and inferring "\
                    " the relationship between several entities within the given text"
        
    # prompt = f"""Paraphrase the sentence '{original_text}'
    # keeping the relationship '{relation_name}' between '{entity1}' and '{entity2}'.
    # Keep the special tokens of the original sentence intact with entities token.
    #  Do not change the entity names."""
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



def new_sentence_generation(original_text, relation_name) -> Prompt:
    header = "Please generate an entire new sentence with a different context while preserving the specified relationship. "\
        	f"Here's a sentence:\n\n {original_text}\n"
    header += f"The words inside the tag e1 and e2 are the two entities present in the text and their relationship is:\n{relation_name}\n"

    print(original_text, relation_name)
    if relation_name == 'Other':
        header += ''
    elif relation_name.index('e1') < relation_name.index('e2'):
        header += "In this relationship, the two words separated by - denotes the relationship direction from entity e1 to e2 in the given sentence"
    else:
        header += "In this relationship, the two words separated by - denotes the relationship direction from entity e2 to e1 in the given sentence"\
    
    if relation_name != 'Other':
        footer = "Please output the new sentence by keeping the entities and their relationship intact, meaning the word in entities should be exactly same "\
            f"as given: {relation_name}. The direction flow of relation should be maintained in the output and the new sentence should be inside [CLS] [SEP] as given above."
    else:
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

def generate_pseudo_label(original_text, predefined_relations) -> Prompt:
    header = f"Analyze the given sentence carefully and determine how the entities inside the tags e1 and e2 are related. "\
        f"Here's a sentence:\n\n {original_text}\n"
    
    
    footer = "Please output the relationship between entities in the format below:\n <relation> </relation>\n "\
            f"Select only one option from the predefined relations list given in the list below: \n{predefined_relations}\n" \
                "The relationship in the list denote the flow of relation from entity e1 to e2 or e2 to e1. "\
            "Also, output the confidence score (0-1) indicating your certainty about the relationship inside the <confidence> tag."
    
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

def new_sentence_generation_rem(original_text, relation_name) -> Prompt:
    header = "Please generate an entire new sentence with a different context while preserving the specified relationship. "\
        	f"Here's a sentence:\n\n {original_text}\n"
    header += f"The words inside the tag e1 and e2 are the two entities present in the text and their relationship is:\n{relation_name}\n"

    print(original_text, relation_name)
    if relation_name.index('e1') < relation_name.index('e2'):
        header += "In this relationship, the two words separated by - denotes the relationship direction from entity e1 to e2 in the given sentence"
    else:
        header += "In this relationship, the two words separated by - denotes the relationship direction from entity e2 to e1 in the given sentence"\
            
    footer = "Please output the new sentence by keeping the entities and their relationship intact, meaning the word in entities should be exactly same "\
            f"as given: {relation_name}. The direction flow of relation should be maintained in the output and the new sentence should be inside [CLS] [SEP] as given above."
        
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
        