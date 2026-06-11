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
    

class DietDataset(Dataset):

    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, seq_len, sample_attribute_contains=False):
        self.vocab = vocab
        self.attribute = attribute
        self.seq_len = seq_len
        self.sample_attribute_contains = sample_attribute_contains

        self.corpus_path = corpus_path

        df = pd.read_csv(corpus_path, engine='python')
        self.corpus = df.to_numpy().tolist() #list

        self.incidence_matrix = create_diet_incidence_matrix(self.corpus, self.vocab)
        self.init_n_corpus = len(self.corpus)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, seq_idx):
        '''
        encoder_input: encoder에 들어갈 masking 된 sequence
        decoder_input: decoder에 들어갈 <s>, </s> token을 포함하는 target sequence
        encoder_label: masking 된 token의 위치에만 original token
        '''
        original = self.corpus[seq_idx].copy()
        decoder_target = self.add_special_token(original)
        attribute_mask = self.get_attribute_mask(original)
        encoder_input, mask_label, mask_idx = self.random_word(original)
        '''
        encoder_input: masking 자리에는 random한 token이 들어감
        mask_label: sequence mask label
        mask_index: sequence의 mask된 위치 idx
        '''
        random_attribute_mask = self.get_random_attribute_mask(mask_idx, attribute_mask, self.sample_attribute_contains)
        #decoder_target = self.add_special_token(original)

        output = {"encoder_input": encoder_input,
                  "encoder_label": mask_label, #"segment_label": segment_label,
                  "decoder_target": decoder_target,
                  "attribute_mask" : attribute_mask,
                  "random_attribute_mask" : random_attribute_mask}

        return {key: torch.tensor(value) for key, value in output.items()}

    def add_corpus(self, new_data):
        # batch 만큼 넣어줘야함 2-array list 형식으로
        print("num of cadidates to add: ", len(new_data))
        past_len = len(self.corpus)
        self.corpus.extend(new_data)
        #self.corpus = remove_duplicates_2d_list(self.corpus) # queue 처럼 기존 데이터가 없어지게 하는 방법 추가?
        current_len = len(self.corpus)
        print("num of added sample w/o duplicate: ", current_len-past_len)

    def update_incidence_matrix(self):
        self.incidence_matrix = create_diet_incidence_matrix(self.corpus, self.vocab)
        
    def add_special_token(self, original):
        added = ['<s>']+original+['</s>']
        return list(map(self.vocab.convert_tokens_to_ids, added))

    def random_word(self, tokens):
        output_label = []
        masking_idx = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 20% randomly change token to current token
                if prob < 0.2:
                    tokens[i] = self.vocab.convert_tokens_to_ids(token)

                # 80% randomly change token to random token
                elif prob < 1.0:
                    tokens[i] = random.randrange(self.vocab.vocab_size)

                else:
                    pass

                output_label.append(self.vocab.convert_tokens_to_ids(token))
                masking_idx.append(i)
            else: #(15%)
                tokens[i] = self.vocab.convert_tokens_to_ids(token)
                output_label.append(0)

        return tokens, output_label, masking_idx

    def get_attribute_mask(self, original):
        attribute_mask =  [self.attribute.attribute_codebook[token] for token in original] # [seq_len, num_attribute]
        mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))
        mask = mlb.fit_transform(attribute_mask)

        return mask

    def get_random_attribute_mask(self, sequence_mask_idx, attribute_mask, sample_attribute_contains): # [seq_len, num_attribute]
        # masked token에 대해서만 random attribute를 부여

        random_attribute = copy.deepcopy(attribute_mask) # 1d list
        num_attributes = self.attribute.attribute_size
        if sample_attribute_contains == False: 
            '''해당 방법은 attribute을 추가한 token을 예측하는 걸 훨씬 잘하게 학습함'''
            for idx in sequence_mask_idx:
                prob = random.random()
                if prob > 0.1: 
                    prob /= 0.9
                    # 특정 1~3개의 negative 조합 # 특정 1~2개의 positive 조합

                    n_p_condition = np.random.poisson(lam=2, size =2)
                    n_p_condition = np.where(n_p_condition ==0, 1, n_p_condition)

                    neg_condition = np.random.choice(num_attributes, n_p_condition[0], replace=False).tolist()
                    pos_condition = np.random.choice(num_attributes, n_p_condition[1], replace=False).tolist()
                    
                    if prob > 0.4: # 40% only neg
                        random_attribute[idx][neg_condition] = 0

                    elif prob > 0.8: #80% only pos
                        random_attribute[idx][pos_condition] = 1

                    else: # 20% neg & pos
                        random_attribute[idx][neg_condition] = 0
                        random_attribute[idx][pos_condition] = 1
                else:
                    # binomial distibution으로 샘플링
                    # 3000개의 attribute이므로, 
                    # 그 중 99.5%의 확률로 forbidden mask 생성 해당 attribute에 0
                    # 그 중 0.5%의 확률로 must_have mask 생성 해당 attribute에 1
                    random_attribute[idx] = np.random.binomial(n=1, p=0.005, size=num_attributes).tolist()
                    
        else:
            # token attribute sampling:
            # negative: token이 가진 attribute 중에서 제거 -> 
            # 하나만 가지는 경우 Random attribute이 아예 없는 attribute가 되기 때문에, 
            # 기존의 attribute 수보다 작은 값을 ranodm sampling (negative 조합)
            for idx in sequence_mask_idx:
                
                token_attribute = [i for i,v in enumerate(random_attribute[idx]) if v==1]
                non_token_attribute = [i for i,v in enumerate(random_attribute[idx]) if v==0]
                
                prob = random.random()
                if prob > 0.1:
                    prob /= 0.9
                
                    n_p_condition = np.random.poisson(lam=2, size =2)
                    
                    if len(token_attribute) == 0: #empty token
                        n_p_condition[0] = 0
                        
                        pos_condition = np.random.choice(non_token_attribute, n_p_condition[1], replace=False).tolist()
                        
                    elif n_p_condition[0] >= len(token_attribute):
                        n_p_condition[0] = np.random.choice(len(token_attribute), replace=True)
                    
                        n_p_condition = np.where(n_p_condition==0, 1, n_p_condition)
                            
                        neg_condition = np.random.choice(token_attribute, n_p_condition[0], replace=False).tolist()
                        pos_condition = np.random.choice(non_token_attribute, n_p_condition[1], replace=False).tolist()
                        
                    else:
                        n_p_condition = np.where(n_p_condition==0, 1, n_p_condition)
                        
                        neg_condition = np.random.choice(token_attribute, n_p_condition[0], replace=False).tolist()
                        pos_condition = np.random.choice(non_token_attribute, n_p_condition[1], replace=False).tolist()
                    
                    if prob > 0.4: # 40% only neg
                        if len(token_attribute) <= 1:
                            pass
                        else:
                            random_attribute[idx][neg_condition] = 0

                    elif prob > 0.8: #80% only pos
                        random_attribute[idx][pos_condition] = 1

                    else: # 20% neg & pos
                        if len(token_attribute) <= 1:
                            random_attribute[idx][pos_condition] = 1
                        else:
                            random_attribute[idx][neg_condition] = 0
                            random_attribute[idx][pos_condition] = 1
                    
                else:
                    random_attribute[idx] = np.random.binomial(n=1, p=0.005, size=num_attributes).tolist()

        return random_attribute



class DietDataset_with_RP(DietDataset):
    def __init__(self, corpus_path, vocab: Vocab, attribute: Attribute, seq_len, max_length, sample_attribute_contains):
        self.vocab = vocab
        self.attribute = attribute
        self.seq_len = seq_len
        df = pd.read_csv(corpus_path, engine='python')
        self.initial_data = df.to_numpy()
        self.max_length = max_length
        self.data_shape = seq_len
        self.sample_attribute_contains = sample_attribute_contains
        
        self.additional_data = np.empty((max_length - len(self.initial_data), seq_len), dtype=object)
        self.add_position = 0
        self.current_length = len(self.initial_data)
        self.init_n_corpus = len(self.initial_data)
        self.incidence_matrix = create_diet_incidence_matrix(df.to_numpy().tolist(), self.vocab)
        self.data_count = 0

    def __len__(self):
        return self.current_length
    
    def __getitem__(self, idx):
        if idx < len(self.initial_data):
            sample =  self.initial_data[idx].tolist()
        else:
            idx -= len(self.initial_data)
            if idx < self.current_length - len(self.initial_data):
                sample = self.additional_data[idx].tolist()
            else:
                raise IndexError("Index out of range")

        original = sample.copy()
        decoder_target = self.add_special_token(original)
        attribute_mask = self.get_attribute_mask(original)
        encoder_input, mask_label, mask_idx = self.random_word(original)

        random_attribute_mask = self.get_random_attribute_mask(mask_idx, attribute_mask, self.sample_attribute_contains)

        output = {"encoder_input": encoder_input,
                  "encoder_label": mask_label, #"segment_label": segment_label,
                  "decoder_target": decoder_target,
                  "attribute_mask" : attribute_mask,
                  "random_attribute_mask" : random_attribute_mask}

        return {key: torch.tensor(value) for key, value in output.items()}

    def get_all_data(self):
        return np.vstack((self.initial_data, self.additional_data)).tolist()

    def add_data(self, new_data_batch):
        past_size = len(new_data_batch)
        print("num of cadidates to add: ", past_size)
        new_data_batch = np.array(new_data_batch, dtype=object)

        # 중복 제거
        all_data = np.vstack((self.initial_data, self.additional_data[:self.add_position]))

        # 중복 검사
        non_duplicate_indices = ~np.any(np.all(new_data_batch[:, None] == all_data, axis=2), axis=1)
        unique_new_data = new_data_batch[non_duplicate_indices]

        num_new = len(unique_new_data)
        self.data_count += num_new

        print("num of added sample w/o duplicate: ", num_new)
        space_left = len(self.additional_data) - self.add_position
        # 순환 저장 로직
        if num_new > 0:
            if num_new <= space_left:
                self.additional_data[self.add_position:self.add_position + num_new] = unique_new_data
            else:
                end_pos = space_left
                self.additional_data[self.add_position:] = unique_new_data[:end_pos]
                remaining_new_data = unique_new_data[end_pos:]
                self.additional_data[:len(remaining_new_data)] = remaining_new_data

            self.add_position = (self.add_position + num_new) % len(self.additional_data)
            self.current_length = min(self.max_length, self.current_length + num_new)