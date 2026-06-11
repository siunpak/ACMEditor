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

import random
from collections import deque

class Vocab(object):
    def __init__(self, corpus_path=None, vocab_path=None, menu_path=None, load_vocab = False, expand = False):
        self.vocab_path = vocab_path
        self.corpus_path = corpus_path
        self.menu_path = menu_path
        '''
        0: [MASK] : mask token
        '''
        #self.special_tokens = ["[MASK]"]
        self.special_tokens = ["<s>", "</s>",'empty']
        self.unk_token = "[UNK]"
        #self.mask_token = self.special_tokens[0]
        #self.mask_idx = self.special_tokens.index("[MASK]")
        
        if load_vocab == True:
            assert vocab_path!=None, "Vocab path must be entered."
            self.vocab_list = self._load_vocab()
        
        else:
            assert (corpus_path!=None) and (vocab_path!=None), "Vocab and Corpus Paths must be entered."
            self.vocab_list = self._create_vocab()
            
        if expand == True:
            assert menu_path != None, "Menu path must be entered"
            self.vocab_list = self._expand_vocab()

        self.dictionary= self._create_dictionary()
        
        self.ids_to_tokens = dict([(ids, tok) for tok, ids in self.dictionary.items()])
    
    def _create_vocab(self):
        """create a vocabulary file into a list."""
        df = pd.read_csv(self.corpus_path, engine='python')
        #df = df.map(str.lower)
        vocab = list()
        for i in range(len(df.columns)): # input sequence length is 13
            vocab += df.iloc[:,i].unique().tolist()
            vocab = list(set(vocab))

        vocab = list(sorted(vocab))
        
        with open(self.vocab_path, "w", encoding="utf-8") as writer:
            for token in vocab:
                writer.write(token + "\n")
        
        return vocab

    def _load_vocab(self):
        #"""Loads a vocabulary file into a list."""
        #with open(self.vocab_path, "r", encoding="utf-8") as reader:
        #    tokens = reader.readlines()
        #vocab = [token.rstrip("\n") for token in tokens]
        vocab_df = pd.read_csv(self.vocab_path)
        vocab = list(vocab_df.entity.unique())
        
        return vocab
    
    
    def _expand_vocab(self):
        """Expand vocabulary into whole menu tokens"""
        
        menu_df = pd.read_csv(self.menu_path, engine= 'python')
        expand_vocab = [i for i in menu_df.name.unique().tolist() if i not in self.vocab_list]
        
        vocab = self.vocab_list + expand_vocab
        
        with open(self.vocab_path, "w", encoding = "utf-8") as writer:
            for token in vocab:
                writer.write(token + "\n")
        
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab_list)

    def _create_dictionary(self):

        dictionary = {token: idx  for idx, token in enumerate(self.special_tokens)}
        value_list = list(range(len(self.special_tokens), len(self.special_tokens) + self.vocab_size))
        vocab_dictionary = dict(zip(self.vocab_list, value_list))
        dictionary.update(vocab_dictionary)
        self.vocab_list = list(dictionary.keys())
        return dictionary

    def convert_tokens_to_ids(self, tokens):
        #return self.dictionary.get(tokens, self.dictionary.get(self.unk_token))
        return self.dictionary[tokens]
    
    def convert_ids_to_tokens(self, index):
        #return self.ids_to_tokens.get(index, self.unk_token)
        return self.ids_to_tokens[index]
    
class Attribute(object):
    def __init__(self, vocab: Vocab, allergy_external_path, ingre_external_path, attribute_type):
        self.allegy_df = pd.read_csv(allergy_external_path, engine='python', encoding = 'cp949')
        self.ingredient_df = pd.read_csv(ingre_external_path, engine='python')
        self.vocab = vocab
        #self.vocab_special_tokens = ["[MASK]", "empty"]
        self.vocab_special_tokens = ["<s>", "</s>",'empty']
        #allegy_df = pd.read_csv("./data/알레르기_DB.csv", encoding='euc-kr', engine='python')
        #ingre_df = pd.read_csv("./data/Total_ingredient_DB_simpled.csv", encoding='euc-kr', engine='python')
        self.attribute_list = None
        self.dictionary = self._create_attribte_dictionary(a_type= attribute_type)
        self.attribute_book = self._create_attribute_book(a_type= attribute_type)
        self.attribute_codebook = self._create_attribute_codebook()
        self.ids_to_tokens = dict([(ids, tok) for tok, ids in self.dictionary.items()])


        if attribute_type == 'ingredient':
            self.allergy_dict = {key:idx for idx,key in enumerate(self.allegy_df.columns[1:])}
            self.allergy_ingre_book = self._make_allergy_ingre_dict()
            self.allergy_ingre_codebook = self._create_allergy_ingre_codebook()

    @property
    def attribute_size(self):
        return len(self.attribute_list)
    
    def convert_tokens_to_ids(self, tokens):
        #return self.dictionary.get(tokens, self.dictionary.get(self.unk_token))
        return self.dictionary[tokens]

    def convert_ids_to_tokens(self, index):
        #return self.ids_to_tokens.get(index, self.unk_token)
        return self.ids_to_tokens[index]

    def _create_attribute_book(self, a_type):
        if a_type == "allergy":
            return self._make_food_allergy_dict()
        elif a_type =="ingredient":
            return self._make_food_ingre_dict()
        else:
            return TypeError("The only possible candidates are ['allergy', 'ingredient']")
    
    def _create_attribute_codebook(self):
        codebook = dict()
        for food, a_list in self.attribute_book.items():
            codebook[food] = list(map(self.convert_tokens_to_ids, a_list))
        return codebook

    def _create_allergy_ingre_codebook(self):
        codebook = dict()
        for allergy, a_list in self.allergy_ingre_book.items():
            codebook[allergy] = list(map(self.convert_tokens_to_ids, a_list))
        return codebook

    def _create_attribte_dictionary(self, a_type):
        if a_type == "allergy":
            self.attribute_list = self.allegy_df.columns[1:]
        elif a_type =="ingredient":
            self.attribute_list = list(self.allegy_df['Name'].unique())
        else:
            return TypeError("The only possible candidates are ['allergy', 'ingredient']")
            
        dictionary = {key : idx for idx, key in enumerate(self.attribute_list)}
        return dictionary


    def _make_ingre_dict(self):
        # output : {ingred : [food1, food2, ...]}
        ingre = self.ingredient_df
        ingre_list = list(self.allegy_df['Name'].unique())
        
        return {ingd : list(ingre[ingre['ingredient'] == ingd].name.unique()) for ingd in ingre_list}

    def _make_allergy_dic(self, allergy_true = True):
        # output : {allergy : [food1, food2, ...]}
        allergy = self.allegy_df
        ingre = self.ingredient_df
        allergy_dic = dict()
        gate = 0

        allergy_list = allergy.columns[1:]
        if allergy_true: gate = 1

        for i in allergy_list:
            temp = list(allergy[allergy[i] == gate]['Name'])
            allergy_dic[i] = list(ingre[ingre['ingredient'].isin(temp)].name.unique())
                
        return allergy_dic

    def _make_food_ingre_dict(self):
        # output : {food : [ingred1, ingred2, ...]}
        ingre = self.ingredient_df
        #food_list = list(ingre['name'].unique())
        food_list = [token for token in self.vocab.vocab_list if token not in self.vocab_special_tokens]
        food_dic = {food : list(ingre[ingre.name == food]['ingredient']) for food in food_list}
        for s_token in self.vocab_special_tokens:
            food_dic[s_token] = []
        return food_dic

    def _make_food_allergy_dict(self):
        # output :  {food : [allergy1, allergy2, ...]}
        allergy = self.allegy_df
        ingre = self.ingredient_df

        food_dic = dict()
        
        #food_list = list(ingre['name'].unique())
        food_list = [token for token in self.vocab.vocab_list if token not in self.vocab_special_tokens]
        allergy_list = allergy.columns[1:]
        
        for i in food_list:
            ingres_in_menu = list(ingre[ingre.name == i]['ingredient'])
            temp = allergy[allergy['Name'].isin(ingres_in_menu)][allergy_list]
            allergy_contains = list(temp.loc[:, temp.gt(0).any()].columns)
            food_dic[i] = allergy_contains
        
        for s_token in self.vocab_special_tokens:
            food_dic[s_token] = []
        return food_dic
    
    def _make_allergy_ingre_dict(self):
        # output : {allergy: [ingre1, ingre22, ..., ]}
        
        allergy = self.allegy_df
        
        allergy_ingre_dict = dict()
        allergy_list = allergy.columns[1:]
        for i in allergy_list:
            allergy_contains = list(allergy[allergy[i] == 1].Name.unique())
            allergy_ingre_dict[i] = allergy_contains
            
        return allergy_ingre_dict


class FinetuneDataset(Dataset):

    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, seq_len, allergy_combination_reference_path,
                 attribute_type):
        self.vocab = vocab
        self.attribute = attribute
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.num_attributes = attribute.attribute_size
        self.attribute_type = attribute_type
        
        self.mlb = MultiLabelBinarizer(classes=np.arange(self.num_attributes))
        self.VA_dictionary = create_vocab_attribute_dictionary(vocab, attribute)
    
        df = pd.read_csv(corpus_path, engine='python')
        df = df.dropna(axis=1)
        
        self.initial_data = df.to_numpy()
        
        #self.before = [seq[:13].tolist() for seq in self.initial_data]
        self.after = [seq[13:26].tolist() for seq in self.initial_data]
        #self.allergy = [seq[-1] for seq in self.initial_data]
        
        
        self.corpus = self.after #list
        self.allergy_combinations = self.get_allergy_combinations(allergy_combination_reference_path)
        self.allergy_labels = {tuple(item): idx for idx, item in enumerate(self.allergy_combinations)}
        self.reverse_allergy_labels = {v:k for k,v in self.allergy_labels.items()}
        
    def get_allergy_labels(self, allergies):
        item_tuple = tuple(allergies)
        return self.allergy_labels.get(item_tuple)
    
    def get_reverse_allergy_labels(self, label):
        return list(self.reverse_allergy_labels.get(label))

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
        decoder_target: decoder에 들어갈 <s>, </s> token을 포함하는 target sequence [batch, seq_len+2]
        encoder_label: masking 된 token의 위치에만 original token [batch, seq_len]
        original_attribute : 기존 token의 attribute mask [batch, seq_len, attribute_size]
        changed_attribute: allergy attribute mask [batch, seq_len, attribute_size]
        target_condition: allergy 조합을 포함하는 attribute masking [batch, attribute]
        '''
        
        '''
        before diet -> 수정 전 식단
        after_diet -> 수정 후 식단
        1) before diet이 target condition을 포함한다면, random masking X, Attribute masking 수행
        2) before diet이 target condition을 포함하지 않으면, random masking O, Attribute masking 수행 X
        '''
        
        original = self.initial_data[seq_idx].copy()
        
        before_diet = original[:13].tolist().copy()
        after_diet = original[13:26].tolist().copy()
        allergy_info = original[-1].split(",").copy()
        
        decoder_target = self.add_special_token(after_diet, allergy_info)
        encoder_input = torch.tensor(self.tokenize(before_diet))
        # Allergy 있는 것에 1을 주고 있음
        target_condition = self.get_condition(allergy_info)
        allergy_info_int = self.get_allergy_labels(allergy_info)
        
        #if sum(target_condition) == 0:
        #    original_attribute = self.get_attribute_mask_idx(before_diet)
        #    encoder_input, problem_mask, problem_idx = self.random_word(before_diet)
        #    attributes_mask = original_attribute.copy()
            
        #else:
            # Allergy를 가지는 token 위치 표시 
        problem_mask, problem_idx, original_attribute = self.examine_and_mask(before_diet, target_condition) # sequence level allergy checker
        
        problem_mask = torch.tensor([0]*13)
        problem_mask[problem_idx] = torch.tensor(self.tokenize(after_diet))[problem_idx]
        
        attributes_mask = original_attribute.copy()
    
        for idx in problem_idx:
            attributes_mask[idx][np.where(target_condition == 1)[0]] = 0
            # 기존 Attribute 중에서 target_condition에 속하는 attribute masking

        output = {"encoder_input": encoder_input.numpy(), #[seq_len]
                  "encoder_label": problem_mask.numpy(), #[seq_len] -> 없어야할 재료를 포함하면 1 
                  "decoder_target": decoder_target,#list(map(self.vocab.convert_tokens_to_ids, ['<s>'])),
                  "original_attribute" : original_attribute,
                  "changed_attribute" : attributes_mask,
                  "target_condition": target_condition,
                  "allergy_info": allergy_info_int}

        return {key: torch.tensor(value) for key, value in output.items()}#, random_allergy_condition
    
    def random_word(self, tokens):
        output_label = []
        masking_idx = []
        seq_prob = random.random()
        if seq_prob < 0.5:
            output_label = [0]*13
            output_tokens = self.tokenize(tokens)
            masking_idx = [0]*13
        else:
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.15:#(15%)
                    prob /= 0.15

                    # 20% randomly change token to current token
                    #if prob < 0.2:
                    #    tokens[i] = self.vocab.convert_tokens_to_ids(token)

                    # 80% randomly change token to random token
                    if prob < 1.0:
                        tokens[i] = random.randrange(self.vocab.vocab_size)
                    else:
                        pass
                    output_label.append(self.vocab.convert_tokens_to_ids(token))
                    masking_idx.append(i)
                else: #(85%)
                    tokens[i] = self.vocab.convert_tokens_to_ids(token)
                    output_label.append(0)
                    
            output_tokens = tokens

        return torch.tensor(output_tokens), torch.tensor(output_label), masking_idx

    def tokenize(self, original):
        return list(map(self.vocab.convert_tokens_to_ids, original))
    
    def get_allergy_tokens(self, target_condition):
        
        allergy_tokens = [key for key, values in self.attribute.attribute_codebook.items() if any(value in np.where(target_condition == 1)[0] for value in values)]
        
        token_idx = self.tokenize(allergy_tokens)
        mlb = MultiLabelBinarizer(classes=np.arange(self.vocab.vocab_size))
        token_map = mlb.fit_transform([token_idx])
        
        return token_map[0]
    
    def add_special_token(self, original, allergy_info):
        added = ['<s>']+original+['</s>'] #+ allergy_info
        #while len(added) < 21:
        #    added = added + ['</s>']
        return list(map(self.vocab.convert_tokens_to_ids, added))

    def get_allergy_combinations(self, allergy_combination_reference_path):

        allergy_combination_df = pd.read_csv(allergy_combination_reference_path, sep=',', header=None)
        allergy_list = allergy_combination_df.to_numpy().tolist()
        temp_allergy_list = []
        for allergy_combination in allergy_list:
            temp = allergy_combination[0].split(',')
            temp_allergy_list.append(temp) 
        #temp_allergy_list = temp_allergy_list + [["0"]]
        return temp_allergy_list
    
    def get_condition(self, allergy_condition): # maksing 되어야하는 attribute에 1
        #random_allergy_conditions = random.choice(self.allergy_combinations)
        #if allergy_condition == ["0"]:
        #    target_condition = [0]*self.attribute.attribute_size
        #else:
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
        self.attribute_type = attribute_type
        
        self.mlb = MultiLabelBinarizer(classes=np.arange(self.num_attributes))
        self.VA_dictionary = create_vocab_attribute_dictionary(vocab, attribute)
    
        df = pd.read_csv(corpus_path, engine='python')
        df = df.dropna(axis=1)
        
        self.initial_data = df.to_numpy()
        self.max_length = max_length
        self.additional_data = np.empty((max_length - len(self.initial_data), seq_len+13+1), dtype=object)
        self.add_position = 0
        self.current_length = len(self.initial_data)
        self.init_n_corpus = len(self.initial_data)
        
        #self.before = [seq[:13].tolist() for seq in self.initial_data]
        self.after = [seq[13:26].tolist() for seq in self.initial_data]
        #self.allergy = [seq[-1] for seq in self.initial_data]
        
        
        self.corpus = self.after #list
        self.allergy_combinations = self.get_allergy_combinations(allergy_combination_reference_path)
        self.allergy_labels = {tuple(item): idx for idx, item in enumerate(self.allergy_combinations)}
        self.reverse_allergy_labels = {v:k for k,v in self.allergy_labels.items()}
        self.data_count = 0
        
    def get_allergy_labels(self, allergies):
        item_tuple = tuple(allergies)
        return self.allergy_labels.get(item_tuple)
    
    def get_reverse_allergy_labels(self, label):
        return list(self.reverse_allergy_labels.get(label))


    def __len__(self):
        return self.current_length
    
    def __getitem__(self, seq_idx):
        
        if seq_idx < len(self.initial_data):
            sample = self.initial_data[seq_idx]
        else:
            seq_idx -= len(self.initial_data)
            if seq_idx < self.current_length - len(self.initial_data):
                sample = self.additional_data[seq_idx]
        '''
        input: Augmented training or 기존 trained data [batch, seq_len]
        label: input seqeunce에서 문제가 된 token (masking) [batch, seq_len]
        attribute mask: 기존 token의 attribute mask [batch, seq_len, attribute_size]
        condition mask: masked token의 문제가 되는 재료를 제거한 attribute mask [batch, seq_len, attribute_size]
        '''
        
        '''
        encoder_input: encoder에 들어갈 masking 된 sequence [batch, seq_len]
        decoder_target: decoder에 들어갈 <s>, </s> token을 포함하는 target sequence [batch, seq_len+2]
        encoder_label: masking 된 token의 위치에만 original token [batch, seq_len]
        original_attribute : 기존 token의 attribute mask [batch, seq_len, attribute_size]
        changed_attribute: allergy attribute mask [batch, seq_len, attribute_size]
        target_condition: allergy 조합을 포함하는 attribute masking [batch, attribute]
        '''
        original = sample.copy()
        
        before_diet = original[:13].tolist().copy()
        after_diet = original[13:26].tolist().copy()
        allergy_info = original[-1].split(",").copy()
        
        decoder_target = self.add_special_token(after_diet, allergy_info)
        encoder_input = torch.tensor(self.tokenize(before_diet))
        target_condition = self.get_condition(allergy_info)
        allergy_info_int = self.get_allergy_labels(allergy_info)
        
        #if sum(target_condition) == 0:
        #    original_attribute = self.get_attribute_mask_idx(before_diet)
        #    encoder_input, problem_mask, problem_idx = self.random_word(before_diet)
        #    attributes_mask = original_attribute.copy()
            
        #else:
            # Allergy 있는 것에 1을 주고 있음
        #target_condition = self.get_condition(allergy_info)
        # Allergy를 가지는 token 위치 표시 
        problem_mask, problem_idx, original_attribute = self.examine_and_mask(before_diet, target_condition) # sequence level allergy checker
        
        problem_mask = torch.tensor([0]*13)
        problem_mask[problem_idx] = torch.tensor(self.tokenize(after_diet))[problem_idx]
        
        attributes_mask = original_attribute.copy()
    
        for idx in problem_idx:
            attributes_mask[idx][np.where(target_condition == 1)[0]] = 0

        output = {"encoder_input": encoder_input.numpy(), #[seq_len]
                  "encoder_label": problem_mask.numpy(), #[seq_len] -> 없어야할 재료를 포함하면 1 
                  "decoder_target": decoder_target,#list(map(self.vocab.convert_tokens_to_ids, ['<s>'])),
                  "original_attribute" : original_attribute,
                  "changed_attribute" : attributes_mask,
                  "target_condition": target_condition,
                  "allergy_info": allergy_info_int}

        return {key: torch.tensor(value) for key, value in output.items()}#, random_allergy_condition

    def random_word(self, tokens):
        output_label = []
        masking_idx = []
        seq_prob = random.random()
        if seq_prob < 0.5:
            output_label = [0]*13
            output_tokens = self.tokenize(tokens)
            masking_idx = [0]*13
        else:
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.15:#(10%)
                    prob /= 0.15

                    # 20% randomly change token to current token
                    #if prob < 0.2:
                    #    tokens[i] = self.vocab.convert_tokens_to_ids(token)

                    # 80% randomly change token to random token
                    if prob < 1.0:
                        tokens[i] = random.randrange(self.vocab.vocab_size)
                    else:
                        pass
                    output_label.append(self.vocab.convert_tokens_to_ids(token))
                    masking_idx.append(i)
                else: #(90%)
                    tokens[i] = self.vocab.convert_tokens_to_ids(token)
                    output_label.append(0)
                    
            output_tokens = tokens

        return torch.tensor(output_tokens), torch.tensor(output_label), masking_idx

    def tokenize(self, original):
        return list(map(self.vocab.convert_tokens_to_ids, original))
    
    def get_allergy_tokens(self, target_condition):
        
        allergy_tokens = [key for key, values in self.attribute.attribute_codebook.items() if any(value in np.where(target_condition == 1)[0] for value in values)]
        
        token_idx = self.tokenize(allergy_tokens)
        mlb = MultiLabelBinarizer(classes=np.arange(self.vocab.vocab_size))
        token_map = mlb.fit_transform([token_idx])
        
        return token_map[0]
    
    def get_all_data(self):
        return np.vstack((self.initial_data, self.additional_data)).tolist()

    
    def add_special_token(self, original, allergy_info):
        added = ['<s>']+original+['</s>']# + allergy_info
        #while len(added) < 21:
        #    added = added + ['</s>']
        return list(map(self.vocab.convert_tokens_to_ids, added))

    def add_data(self, new_data_batch):
        past_size = len(new_data_batch)
        print("num of cadidates to add: ", past_size)
        new_data_batch = np.array(new_data_batch, dtype=object)

        # 중복 제거
        all_data = np.vstack((self.initial_data, self.additional_data[:self.add_position]))

        # 중복 검사
        all_data_set = set(map(tuple, all_data))
        non_duplicate_indices = []
        for i,row in enumerate(new_data_batch):
            if tuple(row) not in all_data_set:
                non_duplicate_indices.append(i)
        #non_duplicate_indices = ~np.any(np.all(new_data_batch[:, None] == all_data, axis=2), axis=1)
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
        #temp_allergy_list = temp_allergy_list + [["0"]]    
        
        return temp_allergy_list
    
    def get_condition(self, allergy_condition): # maksing 되어야하는 attribute에 1
        #random_allergy_conditions = random.choice(self.allergy_combinations)
        #if allergy_condition == ["0"]:
        #    target_condition = [0]*self.attribute.attribute_size
        #else:
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