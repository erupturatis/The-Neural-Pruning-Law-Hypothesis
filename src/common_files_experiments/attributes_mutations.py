from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass
class Mutation:
    field_identified: str
    value_in_field: str
    action: str  # "replace", "remove", or "add"
    replacement_dict: Dict = None  # Used for "replace" or "add"

def mutate_attributes(attributes: List[Dict], mutations: List[Mutation]) -> List[Dict]:
    # copy the attributes
    attributes_copy = attributes.copy()

    for mutation in mutations:
        if mutation.action == "replace":
            for idx, attribute in enumerate(attributes_copy):
                if attribute.get(mutation.field_identified) == mutation.value_in_field:
                    attributes_copy[idx] = mutation.replacement_dict
        elif mutation.action == "remove":
            attributes_copy = [
                attribute for attribute in attributes_copy
                if attribute.get(mutation.field_identified) != mutation.value_in_field
            ]
        elif mutation.action == "add":
            if mutation.replacement_dict:
                attributes_copy.append(mutation.replacement_dict)
    return attributes_copy
