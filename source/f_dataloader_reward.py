from typing import Any
from torch.utils.data import Dataset
import tqdm
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from source.metric import create_diet_incidence_matrix
from source.utils import remove_duplicates_2d_list, create_vocab_attribute_dictionary, merge_without_duplicates_pandas
import copy
from source.f_dataloader_withpair import Vocab, Attribute

import random
from collections import deque

import torch


class FinetuneDataset(Dataset):

    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, seq_len, allergy_combination_reference_path,
                 attribute_type):
        self.vocab = vocab
        self.attribute = attribute
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.num_attributes = attribute.attribute_size
        self.mlb = MultiLabelBinarizer(classes=np.arange(self.num_attributes))
        self.VA_dictionary = create_vocab_attribute_dictionary(vocab, attribute)
        self.attribute_type = attribute_type
    
        df = pd.read_csv(corpus_path, engine='python')
        df = df.dropna(axis=1)
        self.corpus = df.to_numpy().tolist() #list
        self.allergy_combinations = self.get_allergy_combinations(allergy_combination_reference_path)
        self.allergy_labels = {tuple(item): idx for idx, item in enumerate(self.allergy_combinations)}
        self.reverse_allergy_labels = {v:k for k,v in self.allergy_labels.items()}
        #self.incidence_matrix = create_diet_incidence_matrix(self.corpus, self.vocab)
        

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, seq_idx):
        '''
        input: Augmented training or 기존 trained data [batch, seq_len]
        label: input seqeunce에서 문제가 된 token (masking) [batch, seq_len]
        attribute mask: 기존 token의 attribute mask [batch, seq_len, attribute_size]
        condition mask: masked token의 문제가 되는 재료를 제거한 attribute mask [batch, seq_len, attribute_size]
        '''
        
        '''
        encoder_input: encoder에 들어갈 masking 된 sequence [batch, seq_len]
        decoder_target: decoder에 들어갈 <s> token인 target sequence [batch, 1]
        encoder_label: masking 된 token의 위치에만 original token [batch, seq_len]
        original_attribute : 기존 token의 attribute mask [batch, seq_len, attribute_size]
        changed_attribute: allergy attribute mask [batch, seq_len, attribute_size]
        target_condition: allergy 조합을 포함하는 attribute masking [batch, attribute]
        '''

        original = self.corpus[seq_idx].copy()
        
        idx_original = torch.tensor(self.tokenize(original))
        
        target_condition, allergy_info = self.get_random_condition()
        # 있는 것에 1을 주고 있음

        problem_mask, problem_idx, original_attribute = self.examine_and_mask(original, target_condition) # sequence level allergy checker
        #original_attributes_mask = copy.deepcopy(attributes_mask)
        # condition 선반영
        attributes_mask = original_attribute.copy()
        
        for idx in problem_idx:
            attributes_mask[idx][np.where(target_condition == 1)[0]] = 0
            
        allergy_token_map = self.get_allergy_tokens(target_condition)
        
        allergy_info_int = self.get_allergy_labels(allergy_info)
            

        output = {"encoder_input": idx_original.numpy(), #[seq_len]
                  "encoder_label": problem_mask.numpy(), #[seq_len] -> 없어야할 재료를 포함하면 1 
                  "decoder_target": list(map(self.vocab.convert_tokens_to_ids, ['<s>'])),
                  "original_attribute" : original_attribute,
                  "changed_attribute" : attributes_mask,
                  "target_condition": target_condition,
                  "allergy_token_map": allergy_token_map,
                  "allergy_info" : allergy_info_int}

        return {key: torch.tensor(value) for key, value in output.items()}
    
    def get_allergy_labels(self, allergies):
        item_tuple = tuple(allergies)
        return self.allergy_labels.get(item_tuple)
    
    def get_reverse_allergy_labels(self, label):
        return list(self.reverse_allergy_labels.get(label))

    def tokenize(self, original):
        return list(map(self.vocab.convert_tokens_to_ids, original))
    
    def get_allergy_tokens(self, target_condition):
        
        allergy_tokens = [key for key, values in self.attribute.attribute_codebook.items() if any(value in np.where(target_condition == 1)[0] for value in values)]
        
        token_idx = self.tokenize(allergy_tokens)
        mlb = MultiLabelBinarizer(classes=np.arange(self.vocab.vocab_size))
        token_map = mlb.fit_transform([token_idx])
        
        return token_map[0]
    
    def add_special_token(self, original):
        added = ['<s>']+original+['</s>']
        return list(map(self.vocab.convert_tokens_to_ids, added))

    def get_allergy_combinations(self, allergy_combination_reference_path):

        allergy_combination_df = pd.read_csv(allergy_combination_reference_path, sep=',', header=None)
        allergy_list = allergy_combination_df.to_numpy().tolist()
        temp_allergy_list = []
        for allergy_combination in allergy_list:
            temp = allergy_combination[0].split(',')
            temp_allergy_list.append(temp)
        return temp_allergy_list
    
    def get_random_condition(self): # maksing 되어야하는 attribute에 1
        random_allergy_conditions = random.choice(self.allergy_combinations)
        
        if self.attribute_type == 'ingredient':
            allergy_mask = [self.attribute.allergy_ingre_codebook[allergy] for allergy in random_allergy_conditions]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform(allergy_mask)
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1) #[attribute_size]
            
        else:
            allergy_mask = [self.attribute.dictionary[allergy] for allergy in random_allergy_conditions]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform([allergy_mask])
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1)
        
        return target_condition, random_allergy_conditions
    
        
    def get_attribute_mask_idx(self, original):
        attribute_mask =  [self.attribute.attribute_codebook[token] for token in original] # [seq_len, num_attribute]
        mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
        mask = mlb.fit_transform(attribute_mask)

        return mask
    
    #def get_attribute_mask_idx(self, original):
    #    attribute_id_list =  [self.VA_dictionary[token] for token in original] # [seq_len, num_attribute]
    #    return self._get_attribute(attribute_id_list)  
    
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


class FinetuneDataset_with_RP(Dataset):
    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, seq_len, max_length, allergy_combination_reference_path,
                 attribute_type):
        self.vocab = vocab
        self.attribute = attribute
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.num_attributes = attribute.attribute_size
        self.mlb = MultiLabelBinarizer(classes=np.arange(self.num_attributes))
        self.VA_dictionary = create_vocab_attribute_dictionary(vocab, attribute)
        self.attribute_type = attribute_type
    
        df = pd.read_csv(corpus_path, engine='python')
        df = df.dropna(axis=1)
        self.corpus = df.to_numpy()
        self.max_length = max_length
        self.allergy_combinations = self.get_allergy_combinations(allergy_combination_reference_path)
        self.allergy_labels = {tuple(item): idx for idx, item in enumerate(self.allergy_combinations)}
        self.reverse_allergy_labels = {v:k for k,v in self.allergy_labels.items()}
        #self.incidence_matrix = create_diet_incidence_matrix(self.corpus, self.vocab)
        
        self.additional_data = np.empty((max_length - len(self.corpus), seq_len), dtype=object)
        self.add_position = 0
        self.current_length = len(self.corpus)
        self.init_n_corpus = len(self.corpus)
        
        self.data_count = 0
        
    def get_allergy_labels(self, allergies):
        item_tuple = tuple(allergies)
        return self.allergy_labels.get(item_tuple)
    
    def get_reverse_allergy_labels(self, label):
        return list(self.reverse_allergy_labels.get(label))

    def __len__(self):
        return self.current_length
    
    def __getitem__(self, idx):
        if idx < len(self.corpus):
            sample =  self.corpus[idx].tolist()
        else:
            idx -= len(self.corpus)
            if idx < self.current_length - len(self.corpus):
                sample = self.additional_data[idx].tolist()
            else:
                raise IndexError("Index out of range")
        
        original = sample.copy()
        idx_original = torch.tensor(self.tokenize(original))
        
        target_condition, allergy_info = self.get_random_condition()
        # 있는 것에 1을 주고 있음

        problem_mask, problem_idx, original_attribute = self.examine_and_mask(original, target_condition) # sequence level allergy checker
        #original_attributes_mask = copy.deepcopy(attributes_mask)
        # condition 선반영
        attributes_mask = original_attribute.copy()
        
        for idx in problem_idx:
            attributes_mask[idx][np.where(target_condition == 1)[0]] = 0
            
        allergy_token_map = self.get_allergy_tokens(target_condition)
        
        allergy_info_int = self.get_allergy_labels(allergy_info)
            

        output = {"encoder_input": idx_original.numpy(), #[seq_len]
                  "encoder_label": problem_mask.numpy(), #[seq_len] -> 없어야할 재료를 포함하면 1 
                  "decoder_target": list(map(self.vocab.convert_tokens_to_ids, ['<s>'])),
                  "original_attribute" : original_attribute,
                  "changed_attribute" : attributes_mask,
                  "target_condition": target_condition,
                  "allergy_token_map": allergy_token_map,
                  "allergy_info": allergy_info_int}

        return {key: torch.tensor(value) for key, value in output.items()}#, random_allergy_condition

    def tokenize(self, original):
        return list(map(self.vocab.convert_tokens_to_ids, original))
    
    def get_allergy_tokens(self, target_condition):
        
        allergy_tokens = [key for key, values in self.attribute.attribute_codebook.items() if any(value in np.where(target_condition == 1)[0] for value in values)]
        
        token_idx = self.tokenize(allergy_tokens)
        mlb = MultiLabelBinarizer(classes=np.arange(self.vocab.vocab_size))
        token_map = mlb.fit_transform([token_idx])
        
        return token_map[0]
    
    def get_all_data(self):
        return np.vstack((self.corpus, self.additional_data)).tolist()
    
    def add_special_token(self, original):
        added = ['<s>']+original+['</s>']
        return list(map(self.vocab.convert_tokens_to_ids, added))

    def add_data(self, new_data_batch):
        past_size = len(new_data_batch)
        print("num of cadidates to add: ", past_size)
        new_data_batch = np.array(new_data_batch, dtype=object)

        # 중복 제거
        all_data = np.vstack((self.corpus, self.additional_data[:self.add_position]))

        # 중복 검사
        all_data_set = set(map(tuple, all_data))
        non_duplicate_indices = []
        for i,row in enumerate(new_data_batch):
            if tuple(row) not in all_data_set:
                non_duplicate_indices.append(i)
        unique_new_data = new_data_batch[non_duplicate_indices]
        
        num_new = len(unique_new_data)
        self.data_count += num_new
        
        print("num of added sample w/o duplicate: ", num_new)
        if num_new > len(self.additional_data):
            unique_new_data = unique_new_data[-len(self.additional_data):]
            num_new = len(self.additional_data)
        space_left = len(self.additional_data) - self.add_position
        
        # 순환 저장 로직
        if num_new > 0:
            if num_new <= space_left:
                self.additional_data[self.add_position:self.add_position + num_new] = unique_new_data
            else:
                end_pos = space_left
                self.additional_data[self.add_position:] = unique_new_data[:end_pos]
                remaining_new_data = unique_new_data[end_pos:]
                #
                remaining_length = min(len(remaining_new_data), self.add_position)
                self.additional_data[:remaining_length] = remaining_new_data[:remaining_length]

            self.add_position = (self.add_position + num_new) % len(self.additional_data)
            self.current_length = min(self.max_length, self.current_length + num_new)


    def get_allergy_combinations(self, allergy_combination_reference_path):

        allergy_combination_df = pd.read_csv(allergy_combination_reference_path, sep=',', header=None)
        allergy_list = allergy_combination_df.to_numpy().tolist()
        temp_allergy_list = []
        for allergy_combination in allergy_list:
            temp = allergy_combination[0].split(',')
            temp_allergy_list.append(temp)
        return temp_allergy_list
    
    def get_random_condition(self): # maksing 되어야하는 attribute에 1
        random_allergy_conditions = random.choice(self.allergy_combinations)
        
        if self.attribute_type == 'ingredient':
            allergy_mask = [self.attribute.allergy_ingre_codebook[allergy] for allergy in random_allergy_conditions]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform(allergy_mask)
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1) #[attribute_size]
            
        else:
            allergy_mask = [self.attribute.dictionary[allergy] for allergy in random_allergy_conditions]
            mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
            mask = mlb.fit_transform([allergy_mask])
            mask = np.any(mask ==1, axis=0).astype(int)
            target_condition = mask.reshape(-1)
        
        return target_condition, random_allergy_conditions
    
        
    def get_attribute_mask_idx(self, original):
        attribute_mask =  [self.attribute.attribute_codebook[token] for token in original] # [seq_len, num_attribute]
        mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
        mask = mlb.fit_transform(attribute_mask)

        return mask
    
    #def get_attribute_mask_idx(self, original):
    #    attribute_id_list =  [self.VA_dictionary[token] for token in original] # [seq_len, num_attribute]
    #    return self._get_attribute(attribute_id_list)  
    
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
