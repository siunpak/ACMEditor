import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import tqdm
import torch
import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import copy

from torch.utils.data import DataLoader

from source.f_dataloader_withpair import Vocab, Attribute# DietDataset

class EditDataset(Dataset):

    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, allergy_condition, attribute_type):
        self.vocab = vocab
        self.corpus_path = corpus_path

        df = pd.read_csv(corpus_path, engine='python')
        df = df.dropna(axis=1)
        #df = df.map(str.lower)
        self.corpus = df.to_numpy().tolist()#list
        self.allergy_condition = allergy_condition
        self.vocab = vocab
        self.attribute = attribute
        self.attribute_type = attribute_type

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, seq_idx):

        original = self.corpus[seq_idx].copy()
        idx_original = torch.tensor(self.tokenize(original))

        target_condition = self.get_condition(self.allergy_condition)
        # 있는 것에 1을 주고 있음

        problem_mask, problem_idx, original_attribute = self.examine_and_mask(original, target_condition) # sequence level allergy checker
        #original_attributes_mask = copy.deepcopy(attributes_mask)
        # condition 선반영
        attributes_mask = original_attribute.copy()
        
        for idx in problem_idx:
            attributes_mask[idx][np.where(target_condition == 1)[0]] = 0
            
        allergy_token_map = self.get_allergy_tokens(target_condition)

        output = {"encoder_input": idx_original.numpy(), #[seq_len]
                  "encoder_label": problem_mask.numpy(), #[seq_len] -> 없어야할 재료를 포함하면 1 
                  "decoder_target": list(map(self.vocab.convert_tokens_to_ids, ['<s>'])),
                  "changed_attribute" : attributes_mask,
                  "target_condition": target_condition,
                  "allergy_token_map": allergy_token_map}

        return {key: torch.tensor(value) for key, value in output.items()}#, random_allergy_condition
    
    def get_condition(self, allergy_condition): # maksing 되어야하는 attribute에 1
    
        if self.attribute_type == 'ingredient':
            allergy_mask = [self.attribute.allergy_ingre_codebook[allergy] for allergy in allergy_condition]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform(allergy_mask)
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1) #[attribute_size]
            
        else:
            allergy_mask = [self.attribute.dictionary[allergy] for allergy in allergy_condition]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform([allergy_mask])
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1)
        
        return target_condition
    
    def tokenize(self, original):
        return list(map(self.vocab.convert_tokens_to_ids, original))
    
    def get_allergy_tokens(self, target_condition):
        
        allergy_tokens = [key for key, values in self.attribute.attribute_codebook.items() if any(value in np.where(target_condition == 1)[0] for value in values)]
        
        token_idx = self.tokenize(allergy_tokens)
        mlb = MultiLabelBinarizer(classes=np.arange(self.vocab.vocab_size))
        token_map = mlb.fit_transform([token_idx])
        
        return token_map[0]
    
    def examine_and_mask(self, sequence, target_condition):
        '''
        # function : sequence내에서 neg_condition, pos_condition을 조사하고 만족하지 못하는 조건이 있는 token에 masking을 씌움
        #  discriminating masked token which not satisfy the conditions
        '''
        original = copy.deepcopy(sequence) # [seq_len]

        original_attributes = self.get_attribute_mask_idx(original) # [seq_len, attribute]

        # negative mask generation : 없어야 하는 재료 idx
        neg_idx = np.where(target_condition == 1)[0]
        
        problem_mask = torch.tensor(np.any(original_attributes[:, neg_idx]==1, axis=1).astype(int))
        #없어야 하는 재료를 가진 부분은 1 (문제가 있는 시퀀스에 1)
        #[seq_len]

        problem_multihot = (problem_mask == True) # [seq_len]
        problem_idx = problem_multihot.nonzero(as_tuple=True)[0] # [problem_index_in_seq_len]

        #problem_token = masked_sequence[problem_row]
        #masked_sequence[problem_row] = self.vocab.mask_idx

        # 문제가 되는 것에 masking. problem_mask에는 문제가 된 token자리에 1이 있음
        return problem_mask, problem_idx, original_attributes
    
    def get_attribute_mask_idx(self, original):
        attribute_mask =  [self.attribute.attribute_codebook[token] for token in original] # [seq_len, num_attribute]
        mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
        mask = mlb.fit_transform(attribute_mask)

        return mask
