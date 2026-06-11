import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.Encoder import AGGBART_Encoder
from model.Decoder import AGGBART_Decoder
from model.BART import AGGBART

from source.dataloader import Attribute, Vocab, DietDataset, DietDataset_with_RP
from source.utils import create_vocab_attribute_dictionary, remove_duplicates_2d_list, save_to_csv
from source.metric import new_mispos_score, check_duplicates
#from ..source.diet_utils import get_composition_reference, get_nutrition_reference, get_ingredient_reference, create_ingredient_cooccurrence_matrix, create_food_cooccurrence_matrix
from source.sampler import TemperatureSampler, NucleusSampler, GreedySampler, TopKSampler
import math
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.tensorboard import SummaryWriter
import tqdm
import copy
import pandas as pd
import random
import torch.optim.lr_scheduler as lr_scheduler

#for wandb
#import wandb

class AAGBARTTrainer:
    """
    BARTTrainer make the pretrained BART model with two LM training method.

    """
    def __init__(self, bart, vocab:Vocab, attribute:Attribute, train_data: DietDataset_with_RP,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), 
                 with_cuda: bool = True, cuda_devices=None, 
                 u_decode="argmax",  sampling = "random", k = 5,
                 food_columns_dict = None, morning_comb=None, lunch_comb=None, 
                 dinner_comb = None):
        """
        :param bart: BART model which you want to train
        :param vocab: total word vocab size
        :param attribute : ~ 
        :param train_data : train datasets
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        """

        # Setup cuda device for BART training, argument -c, --cuda should be true
        #cuda_condition = torch.cuda.is_available() and with_cuda
        import os
        os.environ["CUDA_VISIBLE_DEVICE"]="1"
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.attribute = attribute
        self.vocab = vocab
        self.VA_dictionary = create_vocab_attribute_dictionary(vocab, attribute)

        # Setting the train and test data loader
        self.train_data = train_data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_corpus = train_data.__len__()

        self.td_batch_size = train_dataloader.batch_size
        self.td_num_workers = train_dataloader.num_workers

        self.food_columns_dict = food_columns_dict
        self.morning_comb = morning_comb
        self.lunch_comb = lunch_comb
        self.dinner_comb = dinner_comb


        self.mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))

        self.model = bart.to(self.device)

        if u_decode == "argmax":
            self.decoding = GreedySampler()
        elif u_decode == "nucleus":
            self.decoding = NucleusSampler(p=0.01, sampler= TemperatureSampler(5.))
        elif u_decode == "topk":
            self.decoding = TopKSampler(k=50, sampler= TemperatureSampler(5.))
        else:
            pass



        # Setting the Adam optimizer with hyper-param
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=0.001)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        
        self.writer = SummaryWriter()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def save_data(self, new_sequence_list, file_path): # masked_loc_list
        # dataset update
        num_new_seq = len(new_sequence_list)
        filtered_sequence_list = self.check_sequence_by_domain_knowledge(new_sequence_list) # masked_loc_list
        
        filtered_sequence_list = remove_duplicates_2d_list(filtered_sequence_list)
        num_filtered_seq = len(filtered_sequence_list)
        print("num of added sample : {}/{} ({}%)".format(num_filtered_seq, num_new_seq, round(num_filtered_seq*100/num_new_seq, 3)))
        
        if num_filtered_seq !=0:
            # save dataset
            save_to_csv(file_path, filtered_sequence_list)
            self.train_data.add_data(filtered_sequence_list)

            # dataset update
            '''
            if num_filtered_seq > self.max_agm_num:
                filtered_sequence_list = random.sample(filtered_sequence_list, self.max_agm_num)
            
            self.train_data.add_corpus(filtered_sequence_list)
            self.max_agm_num = int(self.train_data.__len__() * self.agm_ratio)
            '''
            # dataloader update
            self.train_dataloader = DataLoader(self.train_data, batch_size= self.td_batch_size, num_workers= self.td_num_workers, shuffle=True)

    def add_data(self, new_sequence_list, masked_loc_list):
        # dataset update
        num_new_seq = len(new_sequence_list)
        filtered_sequence_list = self.check_sequence_by_domain_knowledge(new_sequence_list, masked_loc_list)
        
        filtered_sequence_list = remove_duplicates_2d_list(filtered_sequence_list)
        num_filtered_seq = len(filtered_sequence_list)
        print("num of added sample : {}/{} ({}%)".format(num_filtered_seq, num_new_seq, round(num_filtered_seq*100/num_new_seq, 3)))
        
        if num_filtered_seq !=0:
            if num_filtered_seq > self.max_agm_num:
                num_filtered_seq = self.max_agm_num
                selected_idx = np.random.choice(range(len(filtered_sequence_list)), num_filtered_seq, replace=False)
                filtered_sequence_list = [filtered_sequence_list[idx] for idx in selected_idx]

            # dataset update
            print("max arg num : ", self.max_agm_num)
            #filtered_sequence_list = [list(map(self.vocab.convert_ids_to_tokens, line)) for line in filtered_sequence_list]
            self.train_data.add_corpus(filtered_sequence_list)
            self.max_agm_num = int(self.train_data.__len__() * self.agm_ratio)
            
            # dataloader update
            self.train_dataloader = DataLoader(self.train_data, batch_size= self.td_batch_size, num_workers= self.td_num_workers)


    def train(self, epoch):
        print("num of corpus :", self.train_data.__len__())
        self.model.train()
        avg_correct = self.iteration(epoch, self.train_dataloader)
        # check new sequences for composition and nutrition from domain knowledge

        return avg_correct
    
    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            avg_correct = self.iteration(epoch, self.test_dataloader, train=False)
            
        return avg_correct

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activate
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        avg_total_loss = 0.0
        avg_recon_loss = 0.0
        #avg_ult_loss = 0.0
        correct_count = 0
        n_element = 0

        new_sequence_list = []

        for i, data in enumerate(data_loader):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            # data: encoder_inpute, encoder_label(masking), decoder_target(with <s>, </s>), attribute_mask, random_attribute_mask
            

            # 1. Reconstruction loss
            model_output_logprobs = self.model.forward(data["encoder_input"], data["decoder_target"][:,:-1], data["attribute_mask"])
            # output: [batch, vocab_size, sequence_length]
            recon_loss = self.criterion(model_output_logprobs.transpose(1, 2), data["decoder_target"][:, 1:])

            loss = recon_loss.sum()
            avg_recon_loss += recon_loss.sum().item()
        
            # # 2. Random attribute conditioned prediction, attribute_lm_logprobs : [batch_size, seq_len, num_vocab]
            # attribute_lm_logprobs = self.model.forward(data["encoder_input"], data["decoder_target"][:, :-1], data["random_attribute_mask"])
            
            # # Decoding, pred_tokens : [batch_size, seq_len]
            # if train:
            #     pred_tokens = self.decoding(attribute_lm_logprobs[:, :-1, :]) 
            # else:
            #     pred_tokens = attribute_lm_logprobs[:, :-1, :].argmax(dim= -1)

            
            # # negative_mask : 없어야 할 재료가 있는 경우 mask 1. (ex : [0,1,0,1,1,1]) : [batch_size*seq_len, 1]
            # # positive_mask : 있어야 할 재료가 없는 경우 mask 1. (ex : [1,0,0,0,0,1]) : [batch_size*seq_len, 1]
            
            # #only_attribute = True 
            # # negative_mask: 없어진 재료가 있는 경우 mask 1 : [batch_size*seq_len, 1]
            # # positive_mask: 추가된 재료가 없는 경우 + 있어야할 재료가 80%이상 없는 경우 mask 1: [batch_size*seq_len, 1]
            
            # if only_attribute:
            #     negative_mask, positive_mask, idx_target = self.check_and_mask(pred_tokens, data["encoder_label"], data["random_attribute_mask"], data["attribute_mask"], only_attribute=True)
            # else:
            #     negative_mask, positive_mask, idx_target = self.check_and_mask(pred_tokens, data["encoder_label"], data["random_attribute_mask"], data["attribute_mask"], only_attribute=False)
            
            # # pred_lprobs: [batch_size*seq_len, 1]
            # pred_lprobs = attribute_lm_logprobs.view(-1, attribute_lm_logprobs.size(2)).gather(1, pred_tokens.view(-1, 1).to(self.device))

            # if self.u_lambda != 0: #unlikelihood training 과정
            #     # -- Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
                
            #     # 둘 중 하나라도 문제가 있으면 mask 1, 문제가 없으면 mask 0. (ex : [1,1,0,1,1,1])
            #     ult_mask = negative_mask | positive_mask
            #     one_minus_probs= torch.clamp((1.0 - pred_lprobs.exp()), min=1e-8)
            #     ult_loss = -torch.log(one_minus_probs)*(ult_mask)
            #     ult_loss = ult_loss.sum()

            #     loss += self.u_lambda*ult_loss 
            #     avg_ult_loss += ult_loss.item()

            # 3. backward and optimization only in train
            
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                #self.lr_scheduler.step()
    
            real = data["encoder_label"]
            idx_real_nonzero = (real > 0)
            pred_mask = torch.mul(idx_real_nonzero, torch.argmax(model_output_logprobs[:, :-1, :].transpose(1, 2), dim=1).detach())
            pred_element = pred_mask[idx_real_nonzero]
            real_element = real[idx_real_nonzero]
            correct_count += (pred_element == real_element).sum().item()
            n_element += len(real_element) # target 데이터를 잘 맞췄는가 ?
            
            avg_total_loss += loss.item()
        
        avg_total_loss /= len(data_loader)
        avg_recon_loss /= len(data_loader)
        #avg_ult_loss /= len(data_loader)
        avg_correct = correct_count*100 / n_element

        print("EP%d_%s, avg_total_loss=" % (epoch, str_code), round(avg_total_loss,4), "recon_loss= {: .4f}".format(avg_recon_loss), "avg_correct={: .3f}".format(avg_correct)) 
        

        return avg_correct#, new_sequence_list
    
    def save(self, epoch, file_path="output/bart_trained.model"):
        """
        Saving the current BART model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "model.ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def best_save(self, epoch, file_path="output/bart_trained.model"):
        """
        Saving the current BART model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "model.best" #% epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Best Model Saved on:" % epoch, output_path)
        return output_path
    
    def check_and_mask(self, pred_tokens, target, random_attribute_mask, original_attribute_mask, only_attribute=False):

        pred_tokens = pred_tokens.view(-1, 1) # [batch_size*seq_len, 1]
        neg_mask = torch.zeros_like(pred_tokens) # [batch_size*seq_len, 1]
        pos_mask = torch.zeros_like(pred_tokens) # [batch_size*seq_len, 1]

        #random_attribute_mask : [batch, seq_len, num_attribute]
        r_attribute = random_attribute_mask.view(-1, random_attribute_mask.shape[2]) # [batch_size*seq_len, num_attribute]
        target = target.view(-1,1) # [batch_size*seq_len, 1]
        
        #original_attribute_mask : [batch, seq_len, num_attribute]
        o_attribute = original_attribute_mask.view(-1, original_attribute_mask.shape[2])

        idx_target = (target > 0).nonzero(as_tuple=True)[0] # [num_target] <- index of target token
        req_attribute = r_attribute[idx_target].bool() # [num_target, num_attribute]
        #target token에 대한 attribute bool값
        
        '''target token의 원래 attribute 값'''
        original_attribute = o_attribute[idx_target].bool()
        
        pred_no_masked_tokens = pred_tokens[idx_target] # [num_target, 1]
        #target token에 대한 예측 token
        # get attributes of predicted token 

        pred_token_attributes = [self.VA_dictionary[token] for token in pred_no_masked_tokens.squeeze(-1).cpu().tolist()]
        #pred_tokens_name = list(map(self.vocab.convert_ids_to_tokens, pred_masked_tokens.squeeze(-1).cpu().tolist()))
        #pred_token_attributes = [self.attribute.attribute_codebook[token] for token in pred_tokens_name]
        
        # -> [num_target, num_attribute]
        
        pred_token_attributes = torch.from_numpy(self.mlb.fit_transform(pred_token_attributes)).to(self.device) # [num_target, num_attribute]
        pred_token_attributes = pred_token_attributes.bool() # [num_target, num_attribute]

        # check condition
        
        #if mask_type=="negative":
        #    condition = ~req_attribute & pred_token_attributes
        #elif mask_type == "positive":
        #    condition = req_attribute & ~pred_token_attributes
        #else:
        #    raise ValueError("input must be 'negative' or 'positive'")
        if only_attribute == False:

            n_condition = ~req_attribute & pred_token_attributes # [num_target, num_attribute] <- 일치할 수 가 없네 이건 전체가 다 맞아야하는 것. 조금더 완화하는 방법이 필요함
            p_condition = req_attribute & ~pred_token_attributes # [num_target, num_attribute]

            rows_with_n_cond = torch.any(n_condition, dim=1) # [num_target] : bool
            rows_with_p_cond = torch.any(p_condition, dim=1) # [num_target] : bool

            neg_selected_row_indices = torch.nonzero(rows_with_n_cond, as_tuple=True)[0] # [num_not_match_tokens] <- num_target의 index
            pos_selected_row_indices = torch.nonzero(rows_with_p_cond, as_tuple=True)[0] # [num_not_match_tokens] <- num_target의 index

            neg_mask[idx_target[neg_selected_row_indices]] = 1 # 없어야 할 재료가 있는 경우 mask 1 # [batch_size*seq_len, 1]
            pos_mask[idx_target[pos_selected_row_indices]] = 1 # 있어야 할 재료가 없는 경우 mask 1 # [batch_size*seq_len, 1]
            
        else:
            erased_attribute = original_attribute & ~req_attribute #없어진 attribute만 True
            added_attribute = ~original_attribute & req_attribute #추가된 attribute만 True
            
            must_have_attribute = self.must_have_attribute(added_attribute, original_attribute, 0.3)
            #original attribute*rate + added_attribute True
            
            n_condition = erased_attribute & pred_token_attributes #[num_target, num_attribute]
            # -> 없어진 attribute가 예측된 attribute에 있는 경우 True
            p_condition = must_have_attribute & ~pred_token_attributes
            # -> 더한 재료 + 원래 재료 일부가 없는 경우
            
            rows_with_n_cond = torch.any(n_condition, dim=1) # [num_target] : bool
            rows_with_p_cond = torch.any(p_condition, dim=1) # [num_target] : bool

            neg_selected_row_indices = torch.nonzero(rows_with_n_cond, as_tuple=True)[0] # [num_not_match_tokens] <- num_target의 index
            pos_selected_row_indices = torch.nonzero(rows_with_p_cond, as_tuple=True)[0] # [num_not_match_tokens] <- num_target의 index

            neg_mask[idx_target[neg_selected_row_indices]] = 1 # 없앤 재료가 있는 경우 mask 1 # [batch_size*seq_len, 1]
            pos_mask[idx_target[pos_selected_row_indices]] = 1 # 더한 재료+원래 재료 일부가 없는 경우 mask 1 # [batch_size*
        
        return neg_mask, pos_mask, idx_target # [batch_size*seq_len, 1]
    
    def must_have_attribute(self, added_attribute, original_attribute, rate):
        target_token_len, attribute_size = original_attribute.shape
        
        true_counts = original_attribute.sum(dim=1).cpu().numpy()
        num_sample = (rate * true_counts).astype(int)
        
        adjusted_attribute = original_attribute.clone().cpu().numpy()
        
        for i in range(target_token_len):
            if num_sample[i] < true_counts[i]:
                true_index = np.where(original_attribute[i].cpu().numpy())[0]
                sampled_indices = np.random.choice(true_index, num_sample[i], replace=False)
                mask = np.zeros(attribute_size, dtype=bool)
                mask[sampled_indices] = True
                adjusted_attribute[i] = mask
                
        original_por_attribute = torch.tensor(adjusted_attribute, dtype=torch.bool).to(original_attribute.device)
        
        return original_por_attribute | added_attribute
    
    def create_full_sequence(self, agm_mask, pred_data, input_data, idx_target):
        '''
        agm_mask : 대체 가능 토큰이면 1, 아니면 0, [batch_size*seq_len, 1]
        pred_data : 예측된 토큰, [batch_size, seq_len]
        input_data : 입력 데이터, [batch_size, seq_len]
        idx_target : masked 토큰의 index [num_target, 1]
        '''
        batch_size, seq_len = input_data.shape
        changed_data = copy.deepcopy(input_data.view(-1,1)) # [batch_size*seq_len,1]
        pred_data = copy.deepcopy(pred_data.view(-1,1)) # [batch_size*seq_len,1]
        pass_idx = torch.zeros_like(changed_data)

        target_agm_mask = agm_mask[idx_target] # [num_target, 1]
        target_pred_tokens = pred_data[idx_target] # [num_target, 1]

        mask_pass_token = (target_agm_mask > 0) # [num_target, 1]
        idx_pass_token = mask_pass_token.nonzero(as_tuple=True)[0] # [num_pass_token, 1]

        changed_data[idx_target[idx_pass_token]] = target_pred_tokens[idx_pass_token] # [num_pass_token, 1]
        pass_idx[idx_target[idx_pass_token]] = 1
        # idx_target[idx_pass_token] <- pass target의 index
        # pass_idx <- 

        changed_data = changed_data.view(batch_size, seq_len) # [batch_size, seq_len]
        pass_idx = pass_idx.view(batch_size, seq_len)

        rows_with_pass = (pass_idx.any(dim=1)).nonzero(as_tuple=True)[0]
        changed_data = changed_data[rows_with_pass]
        pass_idx = pass_idx[rows_with_pass]

        list_type_data = changed_data.detach().cpu().numpy().tolist()

        str_data = [list(map(self.vocab.convert_ids_to_tokens, line)) for line in list_type_data]
        return str_data #, pass_loc_dict

    def _get_masked_location(self, array):
        nonzero_indices = array.nonzero().tolist()
        rows = dict()
        for row, col in nonzero_indices:
            if row in rows:
                rows[row].append(col)
            else:
                rows[row] = [col]
        return list(rows.values())

    def check_sequence_by_domain_knowledge(self, sequences):
        '''
        sequece : 2d list of sequence (string) [num_sequences, seq_len]
        '''
        #passed_sequences = []
        passed_idx = []
        #span_label = 
        pass_count = {1:0, 2:0}
        for s_idx, seq in enumerate(sequences):
            # [1] check duplication (composition 3) <- 식단 내에 중복 음식이 있는 가? (없어야 함)
            has_duplicates = check_duplicates(seq)
            if has_duplicates == False:
                pass_count[1] += 1
                # [2] check location (composition 1) <- 기존 식단 작성 규칙에 위배 되는 가? (0점이어야 함)
                mps_score, _ = new_mispos_score(seq, self.food_columns_dict, self.morning_comb, self.lunch_comb, self.dinner_comb)
                if mps_score == 0:
                    pass_count[2] += 1
                    passed_idx.append(s_idx)
        #print(pass_count)
        return passed_idx
    
    
    def save_best(self, epoch, file_path="output/bart_trained.model/"):
        """
        Saving the current BART model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "model.best"
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("[BEST]EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    