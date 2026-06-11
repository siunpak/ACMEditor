import torch
import torch.nn as nn
from source.utils import ensure_path, fix_seed, save_to_csv
from source.f_dataloader_withpair import Vocab, Attribute
from source.f_dataloader_withpair import FinetuneDataset, FinetuneDataset_with_RP
from trainer.fintuner_withpair_with_aug import AAGBARTFINETUNER_WITH_PAIR
from model.Encoder import AGGBART_Encoder
from model.Decoder import AGGBART_Decoder
from model.BART import AGGBART

import pandas as pd
import argparse
import pickle
from torch.utils.data import DataLoader
from source.utils import fix_seed, split_dataset_and_save
from tqdm import tqdm


if __name__ == "__main__":
    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset", type=str, default="./data/new/train_pair_diet.csv", help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="./data/new/val_pair_diet.csv", help="test set for evaluate train set")

    # Pretrained model
    parser.add_argument("-data", "--datapath", type=str, default="/home/siun97/Diet/KDD2024/Siun_new_LAB/KDD2025/new_AGGBART/result/new_data/pretrained/ingredient/layer=2/atn_head_16/")

    parser.add_argument("-o", "--output_path",  type=str, default="./result/new_data/with_pretraining_with_aug/", help="ex)output/report.txt")
    parser.add_argument("-allergy", "--allergy_path",  type=str, default="./data/allergy_candidates.csv", help="ex):'./data/AIDIET_....csv',target allergy combination path")
    parser.add_argument("-at", "--attribute_type", default="ingredient", type=str, help="types: ['allergy', 'ingredient']")

    parser.add_argument("-hs", "--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=13, help="maximum sequence len")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help= "dropout ratio")
    
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")

    parser.add_argument("--seed", type=int, default=42, help="set seed value")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    #parser.add_argument("--cuda_device", type=int, default=1, help="CUDA device ids")
    
    parser.add_argument("--tau", type=float, default=1.0, help="temperature in gumble softmax function")
    parser.add_argument("--thres", type=float, default=0.1, help="threshold for the distinct distribution")
    parser.add_argument("--gumble", type=bool, default=False, help="use gumble softmax")

    parser.add_argument("-ula","--u_lambda", type=float, default=0.1, help="lambda for unlikelihood training")
    parser.add_argument("-udm","--u_decode", type=str, default="argmax", help="decoding method for unlikelihood training")
    parser.add_argument("-udg","--u_degrade", type=str, default="False", help="decoding method for unlikelihood training")
    
    
    parser.add_argument("-agm","--agm", type=bool, default=False, help="allow to do data augmentation")
    parser.add_argument("-agx","--agm_x", type=int, default=20, help="setting the number of multiply for the number of augmented data")
    parser.add_argument("-age","--agm_ep", type=int, default=5, help="epoch term for adding augmented data into corpus & number of iteration after satifying the number of augmented data")
    #parser.add_argument("-agr","--agm_rt", type=float, default=0.5, help="ratio to determine the maximum number of augmented data")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    args = parser.parse_args()

    # seed fixing
    fix_seed(args.seed)

    # load vocab, attribute, and reference db
    vocab = Vocab(corpus_path= "./data/train_diet_temp.csv", vocab_path= "./data/entity_vocab_4067.csv",  menu_path="./data/total_nutri_data.csv", load_vocab=True, expand=False)
    print(vocab.vocab_size)
    
    attribute = Attribute(vocab, allergy_external_path = "./data/allergy_db_updated.csv", 
                          ingre_external_path = "./data/total_ingredient_data.csv", 
                          attribute_type = args.attribute_type)
    
    print(len(attribute.attribute_codebook))
    
    print("load vocab and attribute")
    # data load
    train_pair_datapath = args.train_dataset
    test_pair_datapath = args.test_dataset

    #output_path = args.output_path + "test/diff_attn_only_attribute/{}/layer=2/agx{}/ep_{}/uls_{}/atn_head_16/".format(args.attribute_type, args.agm_x, args.agm_ep, args.u_lambda)
    #ensure_path(output_path)
    output_path = args.output_path + "test/"
    ensure_path(output_path)
    
    train_data = FinetuneDataset_with_RP(train_pair_datapath, vocab, attribute, 13, 100000, args.allergy_path, args.attribute_type)
    test_data = FinetuneDataset(test_pair_datapath, vocab, attribute, 13, args.allergy_path, args.attribute_type)

    train_data_loader = DataLoader(train_data, args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_data, 256, num_workers=args.num_workers)
    
    morning_comb = pickle.load(open('./data/morning_comb.pkl', 'rb')) 
    lunch_comb = pickle.load(open('./data/lunch_comb.pkl', 'rb')) 
    dinner_comb = pickle.load(open('./data/dinner_comb.pkl', 'rb')) 
    food_columns_dict = pickle.load(open('./data/food_columns_dict.pkl', 'rb'))
    
    # model
    encoder = AGGBART_Encoder(vocab.vocab_size, attribute.attribute_size, hidden=args.hidden, n_layers=args.layers, 
                              attn_heads=args.attn_heads, dropout=args.dropout,
                              tau=args.tau, thres=args.thres, gumble=args.gumble, alibi=True)
    decoder = AGGBART_Decoder(vocab.vocab_size, hidden=args.hidden,n_layers= args.layers, attn_heads=args.attn_heads, 
                              dropout=args.dropout, tau=args.tau, thres=args.thres, alibi=True)
    #model = AGGBART(encoder, decoder, vocab.vocab_size)
    
    # Fine-tuning 추가 코드
    model_path = args.datapath+"model.best"
    model = torch.load(model_path)
    
    
    print("finish to dataload...Now start to do training")

    trainer = AAGBARTFINETUNER_WITH_PAIR(model, vocab, attribute, train_data, train_data_loader, test_data_loader, args.lr, with_cuda=True,
                                 u_lambda= args.u_lambda, u_decode= args.u_decode, 
                                 u_degrade=args.u_degrade, agm =False, use_attribute_space=False, 
                                 food_columns_dict = food_columns_dict, morning_comb = morning_comb, 
                                 lunch_comb = lunch_comb, dinner_comb = dinner_comb)

    best_correct = 0.0
    
    aug_storage = []
    original_storage = []
    original_allergies = []
    temp_path = output_path + "/augmented_data.csv"
    init_num_corpus = train_data.init_n_corpus
    target_num_corpus = int(init_num_corpus * args.agm_x)
    #final_path = output_path + "/final_augmented_data.csv"
    
    for epoch in tqdm(range(args.epochs)):

        train_correct_ratio_ratio, augmented_data, original_data, original_allergy = trainer.train(epoch)
        
        aug_storage += augmented_data
        original_storage += original_data
        original_allergies += original_allergy 
        print("current data: 1. Augmented_data:{}, 2.original:{}, 3.original_allergy:{}".format(len(augmented_data),
                                                                                                    len(original_data),
                                                                                                    len(original_allergy)))
        
        if (epoch != 0) and (epoch % args.agm_ep == 0):
            trainer.save_data(aug_storage, original_storage, original_allergies, temp_path)
            aug_storage = []
            original_storage = []
            original_allergies = []
        
        if epoch != 0 and epoch % 25 == 0:
            #trainer.save(epoch, output_path)

        if test_data_loader is not None:
            test_avg_correct = trainer.test(epoch)
            if test_avg_correct > best_correct:
                #trainer.save_best(epoch, output_path)
                best_correct = test_avg_correct
                
        num_corpus = trainer.train_data.data_count
        if num_corpus > target_num_corpus:
            print("Reach the target number of dataset...(now just left {} epoch)".format(args.agm_ep))
            for i in range(args.agm_ep):
                trainer.train(epoch+1)
                trainer.test(epoch+1)
            break

    #trainer.save(epoch+1, output_path)
    #save_to_csv(final_path, trainer.train_data.get_all_data())
    
    #split_dataset_and_save(final_path, 0.8)
   # split_dataset_and_save(final_path, 0.9)