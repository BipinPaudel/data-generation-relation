from src.prompts import Prompt

class WaterPrompt():

    def rephrase_generation(self, original_text, relation_name) -> Prompt:
        header = f"Rephrase the following sentence to enhance its complexity while preserving the overall meaning and the " \
                f"specified relationship. Here's a sentence:\n\n {original_text}"
        header += f"Here's a relationship:\n\n{relation_name}\n\n"
        
        
        header += "The given relationship denotes the relationship direction from entity e1 to e2 in the given sentence."
            
            
        footer = "Please output the rephrased sentence while keeping word in entities exactly same "\
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



    def new_sentence_generation(self, original_text, relation_name) -> Prompt:
        header = "Please generate an entire new sentence with a different context while preserving the specified relationship. "\
                f"Here's a sentence:\n\n {original_text}\n"
        header += f"The words inside the tag e1 and e2 are the two entities present in the text and their relationship is:\n{relation_name}\n"


        header += "The given relationship denotes the relationship direction from entity e1 to e2 in the given sentence."\
        
        footer = "Please output a new sentence by keeping the entities and their relationship intact, meaning the word in entities should be exactly same "\
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