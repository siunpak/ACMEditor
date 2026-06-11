import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import FreqDist

def calculate_perplexity(prob, sequence, ignore_index = None):
    '''
    calucate perplexity for composition of sequence
    prob : [batch, seq, dim]
    sequence : [batch, seq]
    ignore_index : ignore to calculate perplexity
    '''
    loss = F.nll_loss(prob, sequence, ignore_index= ignore_index)
    return torch.exp(loss)

def calculate_correctness(n_tokens, pred_tokens:list, neg_conditions:list, pos_conditions:list, attribute):
    '''
    calculate the score whether requirements are met.
    sequence : list of predicted tokens (1d : token level, 2d : sequence level)
    forbidden_list : list of constrained attributes (negative)
    must_have_list : list of constrained attributes (positive)
    attribute : attribute class

    return : {type of level, ratio of forbidden attributes in pred_tokens, ratio of must_have attributes in pred_tokens}
    '''
    try: # discriminator (1d or 2d)
        np.array(pred_tokens)

    except: # sequcnce level evaluation : 2D list
        str_type = "sequence level"
        pred_token_attributes = list()
        for p_list in pred_tokens:
            temp = []
            for token in p_list:
                temp += attribute.attribute_codebook[token]
            pred_token_attributes.append(list(set(temp)))

    else: # token level evaluation : 1D list
        str_type = "token level"
        pred_token_attributes = [attribute.attribute_codebook[token] for token in pred_tokens]

    mlb = MultiLabelBinarizer(classes=np.arange(attribute.attribute_size))
    attributes_multihot = mlb.fit_transform(pred_token_attributes)
    attribute_df = pd.DataFrame(attributes_multihot)

    wrong_neg_idx = (attributes_multihot[:,neg_conditions] == 1).nonzero()[0].tolist()
    #wrong_pos_idx = (attribute_df[pos_conditions] == 0).index.tolist()
    wrong_idx = wrong_neg_idx #+ wrong_pos_idx

    forbidden_ratio_list = ["{}%".format(round(len(attribute_df[attribute_df[column]==1])*100/n_tokens,3)) for column in neg_conditions]
    must_have_ratio_list = ["{}%".format(round(len(attribute_df[attribute_df[column]==1])*100/n_tokens,3)) for column in pos_conditions]
    return {"str" : str_type, "forbid_ratio" : forbidden_ratio_list, "must_have_ratio" : must_have_ratio_list, "wrong_idx" : wrong_idx}

# Calculate Composition score

def calculate_beta_score(sample_diet, incidence_matrix, vocab):

    beta_score = 0 #만점 = sequence 길이(13)
    diet_food_idx = list(map(vocab.convert_tokens_to_ids, sample_diet))
    for index, food_idx in enumerate(diet_food_idx):
        if index in list(np.where(incidence_matrix[food_idx] >= 1)[0]):
            beta_score += 1

    return beta_score

def create_diet_incidence_matrix(diet_data, vocab):

    #이 음식이 이 자리에 나왔던 적이 있느냐?
    incidence_matrix = np.zeros([vocab.vocab_size, len(diet_data[0])])

    for diet in diet_data:
        diet_food_idx = list(map(vocab.convert_tokens_to_ids, diet))
        for index, food_idx in enumerate(diet_food_idx):
            incidence_matrix[food_idx, index] += 1
            
    return incidence_matrix

def calculate_misposition_score(sample_diet, composition_class_dict):
    
    diet_pos = []
    for token in sample_diet:
        if token == '[MASK]' or token == '<s>' or token == '</s>':
            diet_pos.append(token)
        else:
            composition_info = composition_class_dict[token]
            nutri = list(composition_info.index[composition_info == 1].values)     
            diet_pos.append(nutri)
    
    mis_pos = 0
        
    if not any(label in diet_pos[0] for label in ['rice', 'special']):
            mis_pos += 1
    if not any(label in diet_pos[1] for label in ['soup', 'empty']):
            mis_pos += 1
    if not any(label in diet_pos[2] for label in ['side_dish', 'kimchi']):
            mis_pos += 1
    if not any(label in diet_pos[3] for label in ['rice','special']):
            mis_pos += 1
    if not any(label in diet_pos[4] for label in ['soup', 'empty']):
            mis_pos += 1
    if not any(label in diet_pos[5] for label in ['side_dish']):
            mis_pos += 1
    if not any(label in diet_pos[6] for label in ['side_dish', 'empty']):
            mis_pos += 1
    if not any(label in diet_pos[7] for label in ['kimchi']):
            mis_pos += 1
    if not any(label in diet_pos[8] for label in ['rice','special']):
            mis_pos += 1
    if not any(label in diet_pos[9] for label in ['soup', 'empty']):
            mis_pos += 1
    if not any(label in diet_pos[10] for label in ['side_dish']):
            mis_pos += 1
    if not any(label in diet_pos[11] for label in ['side_dish', 'empty']):
            mis_pos += 1
    if not any(label in diet_pos[12] for label in ['kimchi']):
            mis_pos += 1
    
    return mis_pos

def calculate_misposition_score2(sample_diet, composition_class_dict):
    
    diet_pos = []

    diet_pos = []
    for token in sample_diet:
        if token == '[MASK]' or token == '<s>' or token == '</s>':
            diet_pos.append(token)
        else:
            composition_info = composition_class_dict[token]
            nutri = list(composition_info.index[composition_info == 1].values)     
            diet_pos.append(nutri)
    
    mis_pos = 0
    mis_pos_idx = []
    if not any(label in diet_pos[0] for label in ['rice', 'special']):
            mis_pos += 1
            mis_pos_idx.append(0)
    if not any(label in diet_pos[1] for label in ['soup', 'empty']):
            mis_pos += 1
            mis_pos_idx.append(1)
    if not any(label in diet_pos[2] for label in ['side_dish', 'kimchi']):
            mis_pos += 1
            mis_pos_idx.append(2)
    if not any(label in diet_pos[3] for label in ['rice','special']):
            mis_pos += 1
            mis_pos_idx.append(3)
    if not any(label in diet_pos[4] for label in ['soup', 'empty']):
            mis_pos += 1
            mis_pos_idx.append(4)
    if not any(label in diet_pos[5] for label in ['side_dish']):
            mis_pos += 1
            mis_pos_idx.append(5)
    if not any(label in diet_pos[6] for label in ['side_dish', 'empty']):
            mis_pos += 1
            mis_pos_idx.append(6)
    if not any(label in diet_pos[7] for label in ['kimchi']):
            mis_pos += 1
            mis_pos_idx.append(7)
    if not any(label in diet_pos[8] for label in ['rice','special']):
            mis_pos += 1
            mis_pos_idx.append(8)
    if not any(label in diet_pos[9] for label in ['soup', 'empty']):
            mis_pos += 1
            mis_pos_idx.append(9)
    if not any(label in diet_pos[10] for label in ['side_dish']):
            mis_pos += 1
            mis_pos_idx.append(10)
    if not any(label in diet_pos[11] for label in ['side_dish']):
            mis_pos += 1
            mis_pos_idx.append(11)
    if not any(label in diet_pos[12] for label in ['kimchi']):
            mis_pos += 1
            mis_pos_idx.append(12)
    
    return mis_pos, mis_pos_idx

def new_mispos_score(sample_diet, food_columns_dict, morning_comb, lunch_comb, dinner_comb):

    diet_pos = []
    for token in sample_diet:
        if token == '[MASK]' or token == '<s>' or token == '</s>' or token == '밀' or token == '우유' or token == '난류' or token == '대두' or token == '땅콩+견과류':
            diet_pos.append(token)
        else:
            composition_info = food_columns_dict[token]
            diet_pos.append(composition_info)
            
    morning_pos = diet_pos[:3]
    lunch_pos = diet_pos[3:8]
    dinner_pos = diet_pos[8:]

    mis_pos = 0
    mis_pos_idx = []
    if morning_pos not in morning_comb:
        if morning_pos[0] not in list(set([comb[0] for comb in morning_comb])):
            mis_pos += 1
            mis_pos_idx.append(0)
            
        else:
            unique_2 = set(tuple(temp) for temp in [comb[:2] for comb in morning_comb])
            if morning_pos[:2] not in [list(comb) for comb in unique_2]:
                mis_pos += 2
                mis_pos_idx.append(0)
                mis_pos_idx.append(1)
        
            else:
                mis_pos += 3
                mis_pos_idx.append(0)
                mis_pos_idx.append(1)
                mis_pos_idx.append(2)
    else:
        pass
        
    if lunch_pos not in lunch_comb:
        if lunch_pos[0] not in list(set([comb[0] for comb in lunch_comb])):
            mis_pos += 1
            mis_pos_idx.append(3)
        else:
            unique_2 = set(tuple(temp) for temp in [comb[:2] for comb in lunch_comb])
        
            if lunch_pos[:2] not in [list(comb) for comb in unique_2]:
                mis_pos += 2
                mis_pos_idx.append(3)
                mis_pos_idx.append(4)
            else:
                unique_3 = set(tuple(temp) for temp in [comb[:3] for comb in lunch_comb])
                if lunch_pos[:3] not in [list(comb) for comb in unique_3]:
                    mis_pos += 3
                    mis_pos_idx.append(3)
                    mis_pos_idx.append(4)
                    mis_pos_idx.append(5)
                else:
                    unique_4 = set(tuple(temp) for temp in [comb[:4] for comb in lunch_comb])
                    if lunch_pos[:4] not in [list(comb) for comb in unique_4]:
                        mis_pos += 4
                        mis_pos_idx.append(3)
                        mis_pos_idx.append(4)
                        mis_pos_idx.append(5)
                        mis_pos_idx.append(6) 
                    else:
                        mis_pos += 5
                        mis_pos_idx.append(3)
                        mis_pos_idx.append(4)
                        mis_pos_idx.append(5)
                        mis_pos_idx.append(6)
                        mis_pos_idx.append(7)
                        
    else:
        pass 

    if dinner_pos not in dinner_comb:
        if dinner_pos[0] not in list(set([comb[0] for comb in dinner_comb])):
            mis_pos += 1
            mis_pos_idx.append(8)
        else:
            unique_2 = set(tuple(temp) for temp in [comb[:2] for comb in dinner_comb])
            if dinner_pos[:2] not in [list(comb) for comb in unique_2]:
                mis_pos += 2
                mis_pos_idx.append(8)
                mis_pos_idx.append(9)
            else:    
                unique_3 = set(tuple(temp) for temp in [comb[:3] for comb in dinner_comb])
                if dinner_pos[:3] not in [list(comb) for comb in unique_3]:
                    mis_pos += 3
                    mis_pos_idx.append(8)
                    mis_pos_idx.append(9)
                    mis_pos_idx.append(10)
                else:
                    unique_4 = set(tuple(temp) for temp in [comb[:4] for comb in dinner_comb])
                    if dinner_pos[:4] not in [list(comb) for comb in unique_4]:
                        mis_pos += 4
                        mis_pos_idx.append(8)
                        mis_pos_idx.append(9)
                        mis_pos_idx.append(10)
                        mis_pos_idx.append(11) 
                    else:
                        mis_pos += 5
                        mis_pos_idx.append(8)
                        mis_pos_idx.append(9)
                        mis_pos_idx.append(10)
                        mis_pos_idx.append(11)
                        mis_pos_idx.append(12)  
                        
    else:
        pass
    return mis_pos, mis_pos_idx

# Calculate nutrition score

def calculate_nutrition_score(sample_diet, ref_database):
    '''
    calucate the nutrient_score
    sample_diet : target sample diet
    ref_path : reference path
    '''

    nut_standard = nutrient_standards()
    nutri_categories = [
    'Energy', 'Protein', 'Fat', 'Carbohydrate', 'Total Dietary', 'Calcium',
    'Iron', 'Sodium', 'Vitamin A', 'Vitamin B1 (Thiamine)',
    'Vitamin B2 (Rivoflavin)', 'Vitamin C', 'Linoleic Acid',
    'Alpha-Linolenic Acid'
    ]
    menu_sum = pd.DataFrame(columns=nutri_categories)
    
    nutri_sum = np.zeros(14)
    for food in sample_diet:
        
        menu_item = ref_database[food]
        nutri_sum += menu_item.values[0]
    
    menu_sum.loc[0] = nutri_sum
    
    carb_ratio = menu_sum['Carbohydrate'] * 4 / menu_sum['Energy']
    protein_ratio = menu_sum['Protein'] * 4 / menu_sum['Energy']
    fat_ratio = menu_sum['Fat'] * 9 / menu_sum['Energy']
    
    ratio_standards = {
        'Carbohydrate_ratio': (0.55, 0.65),
        'Protein_ratio': (0.07, 0.20),
        'Fat_ratio': (0.15, 0.30)
    }
    
    nutrient_values = {**menu_sum, 'Carbohydrate_ratio': carb_ratio, 'Protein_ratio': protein_ratio, 'Fat_ratio': fat_ratio}
    
    count = 0
    for nutrient, (lower, upper) in {**nut_standard, **ratio_standards}.items():
        is_in_range = check_nutrient_range(nutrient_values[nutrient][0], lower, upper)
        count += is_in_range

    return count

def check_nutrient_range(value, lower, upper):
    """각 영양소별 데이터가 범위 안에 놓이는지 확인"""
    return lower <= value <= upper

def nutrient_standards():
    """영양소 기준치 반환"""

    nutrient_standards = {
        'Energy': (1530, 1870),
        'Protein': (26.25, float('inf')),
        'Total Dietary': (13, float('inf')),
        'Calcium': (525, 2500),
        'Iron': (6.75, 40),
        'Sodium': (0, 1900),
        'Vitamin A': (337.5, 1100),
        'Vitamin B1 (Thiamine)': (0.525, float('inf')),
        'Vitamin B2 (Rivoflavin)': (0.675, float('inf')),
        'Vitamin C': (37.5, 750),
        'Linoleic Acid': (6750, float('inf')),
        'Alpha-Linolenic Acid': (825, float('inf'))
    }
    return nutrient_standards

def get_coherence_score_sequence(diet_sample, npmi_matrix, stem_dict, is_ingre = True):
    
    ### sample_diet : list
    ### masked_loc_list : list ([1,3,4])
    # 해당 token이 자리한 span이 어딘지 확인
    diet_sample = np.array(diet_sample)

    total_coherence = 0

    if is_ingre == True:
        sample_tokens  = []
        for token in diet_sample:
            if token == '</s>' or token == '<s>':
                pass
            else:
                sample_tokens += stem_dict[token]

    else:    
        sample_tokens = diet_sample.copy()

    coherence = 0.0
    for i in range(len(sample_tokens)-1):
        for j in range(i+1, len(sample_tokens)):
            npmi = npmi_matrix.at[sample_tokens[i], sample_tokens[j]]
            if npmi == 0.0:
                return False
            else:
                coherence += npmi
    total_coherence += coherence * (2/(len(sample_tokens)*(len(sample_tokens)-1)))

    return total_coherence/len(diet_sample)


##### Diet Coherence Measure
def get_coherence_score(diet_sample, npmi_matrix, ingredient_dict, masked_loc_list, span_label, is_ingre = True):
    
    ### sample_diet : list
    ### masked_loc_list : list ([1,3,4])
    # 해당 token이 자리한 span이 어딘지 확인
    diet_sample = np.array(diet_sample)
    span_list = [value for idx, value in enumerate(span_label) if idx in masked_loc_list]
    span_label = np.array(span_label)
    total_coherence = 0
    for idx, loc in enumerate(masked_loc_list):
        target_token = diet_sample[loc]
        t_span_name = span_list[idx]
        t_span_idx = np.where(span_label == t_span_name)[0]
        tokens_in_t_span = diet_sample[t_span_idx]

        if is_ingre == True:
            sample_tokens  = []
            for token in tokens_in_t_span:
                sample_tokens += ingredient_dict[token]

        else:    
            sample_tokens = tokens_in_t_span.copy()

        coherence = 0.0
        for i in range(len(sample_tokens)-1):
            for j in range(i+1, len(sample_tokens)):
                npmi = npmi_matrix.at[sample_tokens[i], sample_tokens[j]]
                if npmi == 0.0:
                    return False
                else:
                    coherence += npmi
        total_coherence += coherence * (2/(len(sample_tokens)*(len(sample_tokens)-1)))

    return total_coherence/len(masked_loc_list)


def calculate_npmi(token1, token2, c_xy, p_x, N):
    
    p_xy = c_xy[(token1, token2)]/N
    pxpy = p_x[token1]*p_x[token2]
    pmi = np.log(p_xy/pxpy)
    npmi = pmi/(-np.log(p_xy))
    return npmi

def get_npmi_score(corpus_list, p_x, num_tokens):
    
    tokenized_diets = [set(diet) for diet in corpus_list]
    token_pairs = []

    for diet in tokenized_diets:
        token_pairs += [(token1, token2) for token1 in diet for token2 in diet if token1 != token2]
        
    c_xy = FreqDist(token_pairs) 
    total_npmi = {}
    for pairs in c_xy: 
        total_npmi[pairs] = calculate_npmi(pairs[0], pairs[1], c_xy, p_x, num_tokens)
        
    return total_npmi


def allergy_free_test(sample_diet, ref_database, allergy):
    
    allergy_rate = 0
    allergy_exist = [i for i in sample_diet if ref_database[ref_database.name == i][allergy].values[0].any()]
    allergy_rate += len(allergy_exist)
        
    return allergy_rate

def check_duplicates(diets):
    
    temp_seq = [token for idx, token in enumerate(diets) if (idx not in [0,3,7,8,12]) and token != 'empty']
    #temp_seq = [i for i in temp_seq if i != 'empty']
    has_duplicates = (len(temp_seq) != len(set(temp_seq))) 
    # 있으면 True
    # 없으면 False
    return has_duplicates

def eval_diet(sequences, allergy_reference, allery_name, composition_class_dict, incidence_matrix , nutrition_dict, vocab):
    '''
    sequece : 2d list of sequence (string) [num_sequences, seq_len]
    '''
    total_num_duplicates = 0 # 낮을 수록 좋음
    total_mps_score = 0 # 낮을 수록 좋음
    total_beta_score = 0 # 높을 수록 좋음
    total_rdi_score = 0 # 높을 수록 좋음
    total_success_rate = 0

    for s_idx, seq in enumerate(sequences):
        # [1] check duplication (composition 3) <- 식단 내에 중복 음식이 있는 가? (없어야 함)
        temp_seq = [token for idx, token in enumerate(seq) if idx not in [0,3,7,8,12] ] # 밥 위치 제외
        if len(temp_seq) != len(set(temp_seq)): 
            total_num_duplicates += 1
        # [2] check location (composition 1) <- 기존 식단 작성 규칙에 위배 되는 가? (0점이어야 함)
        mps_score = calculate_misposition_score(seq, composition_class_dict)
        if mps_score != 0:
            total_mps_score += 1

        # [3] check incidence (composition 2) <- 기존 식단의 작성 분포에 위배 되는 가? (13점 만점)
        beta_score = calculate_beta_score(seq, incidence_matrix, vocab)
        total_beta_score += beta_score

        # [4] check nutrition <- 영양학적으로 괜찮은 조합인가? (15점 만점, 기존 식단 평균 점수: 13.33)
        rdi_score = calculate_nutrition_score(seq, nutrition_dict)
        total_rdi_score += rdi_score

        allergy_rate = allergy_free_test(seq, allergy_reference, allery_name)
        if allergy_rate == 0:
            total_success_rate += 1
    
    total_num_duplicates /= len(sequences)
    total_mps_score /= len(sequences)
    total_beta_score /= len(sequences)
    total_rdi_score /= len(sequences)
    total_success_rate /= len(sequences)

    return total_num_duplicates, total_mps_score, total_beta_score,total_rdi_score, total_success_rate