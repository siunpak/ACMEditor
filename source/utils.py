import os
import shutil
import math
import time
import torch
import numpy as np
import random
import csv
import os
import pandas as pd

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(model, name, path):
    torch.save(model.state_dict(), os.path.join(path, name + '.pth'))

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

def create_vocab_attribute_dictionary(vocab, attribute):
    '''
    dictionary : vocab_idx : attribute_idx
    '''
    VA_dictionary = {v_idx : attribute.attribute_codebook[vocab] for vocab, v_idx in vocab.dictionary.items()}

    return VA_dictionary
'''
def remove_duplicates_2d_list(two_dim_list):
    seen = set()
    unique_list = []

    for row in two_dim_list:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_list.append(row)

    return unique_list
'''
def topk(data, num_topk):
    sort, idx = data.sort(descending=False, dim=1)
    return sort[:,:num_topk], idx[:,:num_topk]


def save_to_csv(file_path, data_2d, header=None):
    # 파일 존재 여부와 빈 파일인지 확인
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # 파일 모드 결정 (데이터 추가 또는 새 파일 작성)
    mode = 'a' if file_exists else 'w'

    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)

        # 파일이 새로 생성되었고 헤더가 제공된 경우 헤더 작성
        if not file_exists and header is not None:
            writer.writerow(header)

        # 데이터 작성
        writer.writerows(data_2d)

def remove_duplicates_2d_list(data):
    # DataFrame으로 변환
    df = pd.DataFrame(data)

    # 중복 제거
    unique_df = df.drop_duplicates().reset_index(drop=True)

    # DataFrame을 리스트로 변환
    result = unique_df.values.tolist()

    return result

def merge_without_duplicates_pandas(A, B):
    # DataFrame으로 변환
    df_A = pd.DataFrame(A)
    df_B = pd.DataFrame(B)

    # DataFrame을 합침
    combined_df = pd.concat([df_A, df_B])

    # 중복 제거
    unique_df = combined_df.drop_duplicates().reset_index(drop=True)

    # DataFrame을 리스트로 변환
    result = unique_df.values.tolist()

    return result

def split_dataset_and_save(path, frac):
    df = pd.read_csv(path, header=None,  engine='python')
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    train_df = df.sample(frac=frac, random_state=2023)
    test_df = df.drop(train_df.index)
    train_df.to_csv(path + "train_diet_{}.csv".format(str(frac).split('.')[1]), index=None, columns=None, header=None)
    test_df.to_csv(path + "test_diet_{}.csv".format(str(frac).split('.')[1]), index=None, columns=None, header=None)
