import argparse
import pandas as pd
import torch.nn as nn
from source.utils import ensure_path, save_to_csv
from source.dataloader import Vocab, DietDataset, Attribute, DietDataset_with_RP
from trainer.AAGtrainer import AAGBARTTrainer
from torch.utils.data import DataLoader

from model.Encoder import AGGBART_Encoder
from model.Decoder import AGGBART_Decoder
from model.BART import AGGBART
from source.utils import fix_seed, split_dataset_and_save
import pickle
from tqdm import tqdm



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_normal_(m.weight.data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset", type=str, default="./data/new/train_for_rfm.csv", help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default="./data/new/val_for_rfm.csv", help="test set for evaluate train set")
    
    parser.add_argument("-o", "--output_path",  type=str, default="./result/new_data/pretrained/without_diffatn/", help="ex)output/report.txt")
    parser.add_argument("-allergy", "--allergy_path",  type=str, default="./data/allergy_candidates.csv", help="ex):'./data/AIDIET_....csv',target allergy combination path")
    parser.add_argument("-at", "--attribute_type", default="ingredient", type=str, help="types: ['allergy', 'ingredient']")
    
    parser.add_argument("-hs", "--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=13, help="maximum sequence len")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help= "dropout ratio")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")

    parser.add_argument("--seed", type=int, default=42, help="set seed value")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    
    parser.add_argument("--tau", type=float, default=1.0, help="temperature in gumble softmax function")
    parser.add_argument("--thres", type=float, default=0.1, help="threshold for the distinct distribution")
    parser.add_argument("--gumble", type=bool, default=False, help="use gumble softmax")
    
    parser.add_argument("-udm","--u_decode", type=str, default="argmax", help="decoding method for unlikelihood training")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    args = parser.parse_args()
    
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
    train_datapath = args.train_dataset
    test_datapath = args.test_dataset
    
    output_path = args.output_path + "{}/layer=2/atn_head_16/".format(args.attribute_type)
    ensure_path(output_path)

    train_data = DietDataset_with_RP(train_datapath, vocab, attribute, seq_len=13, max_length=10000,sample_attribute_contains=True)
    test_data = DietDataset(test_datapath, vocab, attribute, seq_len=13, sample_attribute_contains=True)

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
    model = AGGBART(encoder, decoder, vocab.vocab_size)
    
    print("finish to dataload...Now start to do training")
    
    trainer = AAGBARTTrainer(model, vocab, attribute, train_data, train_data_loader, test_data_loader, args.lr, with_cuda=True,
                                 u_decode= args.u_decode, 
                                 food_columns_dict = food_columns_dict, morning_comb = morning_comb, 
                                 lunch_comb = lunch_comb, dinner_comb = dinner_comb)
    
    
    best_correct = 0.0
    
    for epoch in tqdm(range(args.epochs)):
        train_correct_ratio_ratio = trainer.train(epoch) 
        
        if epoch != 0 and epoch % 25 == 0:
            trainer.save(epoch, output_path)

        if test_data_loader is not None:
            test_avg_correct = trainer.test(epoch)
            if test_avg_correct > best_correct:
                trainer.save_best(epoch, output_path)
                best_correct = test_avg_correct
        
    trainer.save(epoch+1, output_path)