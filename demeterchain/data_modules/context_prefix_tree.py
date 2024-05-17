import torch
from typing import List
from transformers.tokenization_utils_base import BatchEncoding


class ContextPrefixTree(object):
    """
    Build a prefix tree of document to limit model generation results
    """

    prefix_tree = None 
    # prefix tree
    input_len = 0 
    # prompt length, it can help differentiate between input and output
    eos_token_id = 2 
    # end of sequence

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def generate_prefix_tree(self, batch_inputs: BatchEncoding, doc_start: int):
        '''
        generate prefix tree for the contexts

        Args:
            batch_inputs: tokenizer output, should set return_offsets_mapping=True
            doc_start: list of context start in each context string, will convert to id start later
        '''
        self.prefix_tree= []
        self.input_len = batch_inputs["input_ids"].size(1)

        for input_ids, offset_mapping in zip(
            batch_inputs["input_ids"], 
            batch_inputs["offset_mapping"], 
        ):
            start, end = self.get_context_id_range(offset_mapping, doc_start) # contxet position in input_ids
            tree = {"position":set(), "sub_tree":{}, "start":start, "end":end} # tree format [[token pos], {next token sub tree}]
            for j, token in enumerate(input_ids[start:end]):
                if token.item() not in tree["sub_tree"]:
                    tree["sub_tree"][token.item()] = {"position":set(), "sub_tree":{}}        
                pos = j + start # token position
                tree["sub_tree"][token.item()]["position"].add(pos) # adding token position to tree
            self.prefix_tree.append(tree)
    
    def add_string_to_prefix_tree(self, input_ids: List[int]): 
        '''
        add string that not in context to prefix_tree, ex:"無法回答". 
        The input_ids is list[int], not string or tensor.
        '''
        for tree in self.prefix_tree:
            for token in input_ids:
                if token not in tree["sub_tree"]:
                    tree["sub_tree"][token] = {"position":set(), "sub_tree":{}}
                tree = tree["sub_tree"][token]

    def get_context_id_range(self, offset_mapping: torch.Tensor, doc_start: int):
        '''
        get contxet position in input_ids.
        should set return_offsets_mapping=True during tokenize
        '''
        start = (offset_mapping[:,0].squeeze(0) >= doc_start).nonzero()[0].item()
        end = (offset_mapping[start:,0]==0).nonzero()
        end = offset_mapping.size(0) if end.size(0) == 0 else start + end[0].item()

        return start, end

    def find_allowed_tokens(self, batch_id: int, input_ids: torch.Tensor):
        '''
        return tokens match the prefix_tree.
        if num_return_sequences > 1 and do_sample=True, the batch_id will cause an error.
        '''
        tree = self.prefix_tree[batch_id]
        end = tree["end"]
        current_token = None
        for token in input_ids[self.input_len:input_ids.size(0)]:
            if token.item() not in tree["sub_tree"]:
                return [self.eos_token_id]
            tree = tree["sub_tree"][token.item()] # go to sub tree
            current_token = token.item()
        
        # generete sub tree
        for pos in tree["position"]:
            if pos + 1 >= end: # eos of the context
                continue
            token = input_ids[pos + 1].item() # get next token
            if token not in tree["sub_tree"]:
                tree["sub_tree"][token] = {"position":set(), "sub_tree":{}}
            tree["sub_tree"][token]["position"].add(pos + 1) # update tree
        
        return list(tree["sub_tree"].keys()) + [self.eos_token_id] # all next token