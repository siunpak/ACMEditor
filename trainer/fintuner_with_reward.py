import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.Encoder import AGGBART_Encoder
from model.Decoder import AGGBART_Decoder
from model.BART import AGGBART
from source.f_dataloader_reward import FinetuneDataset, FinetuneDataset_with_RP

from source.dataloader import Attribute, Vocab, DietDataset
from source.utils import create_vocab_attribute_dictionary, remove_duplicates_2d_list, save_to_csv
from source.metric import new_mispos_score, check_duplicates
from source.sampler import TemperatureSampler, NucleusSampler, GreedySampler, TopKSampler
import math
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import tqdm
import copy
import pandas as pd
import random
from torch.cuda.amp import GradScaler, autocast


class Reward_PPO_trainer:

    def __init__(self, model,
                 vocab:Vocab, attribute:Attribute, train_data: FinetuneDataset_with_RP,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), kl_coef = 0.5, epsilon = 0.2,
                 u_lambda = 0.1, u_decode="argmax", u_degrade=False,
                 agm = True, sampling = "random", k = 5,
                 food_columns_dict = None, morning_comb=None, lunch_comb=None, 
                 dinner_comb = None):


        import os
        os.environ["CUDA_VISIBLE_DEVICE"]="0"
        #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.sampling = sampling
        self.k = k

        self.mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))

        #model setting
        self.old_model = copy.deepcopy(model.to(self.device))
        self.training_model = copy.deepcopy(model.to(self.device))
        self.kl_coef = kl_coef
        self.epsilon = epsilon
        #self.graph_embeddings = graph_embeddings
        self.scaler = GradScaler()
        
        #pretrianedmodel freeze
        for params in self.old_model.parameters():
            params.requires_grad = False

        #training model encoder freeze
        for param in self.training_model.encoder.parameters():
            param.requires_grad = False
            
        self.u_lambda= u_lambda
        self.u_degrade = u_degrade

        if u_decode == "argmax":
            self.decoding = GreedySampler()
        elif u_decode == "nucleus":
            self.decoding = NucleusSampler(p=0.95, sampler= TemperatureSampler(5.))
        elif u_decode == "topk":
            self.decoding = TopKSampler(k=50, sampler= TemperatureSampler(5.))
        else:
            ValueError("options : [argmax, nucleus, topk]")

        self.agm = agm

        # Setting the Adam optimizer with hyper-param
        self.optimizer = optim.Adam(self.training_model.parameters(), lr=lr, betas=betas, weight_decay=0.01)#0.001)
        self.criterion = nn.NLLLoss(ignore_index=-1)
        self.kl_loss = torch.nn.KLDivLoss(reduction = "batchmean", log_target=True)

        self.writer = SummaryWriter()

        self.train_step = 0
        self.test_step = 0

        print("Total Parameters:", sum([p.nelement() for p in self.training_model.parameters()]))
    

    def train(self, epoch):
        print("num of corpus :", self.train_data.__len__())
        self.training_model.train()
        new_seq, avg_reward = self.iteration(epoch, self.train_dataloader)
        #if self.u_degrade=="True":
        #    self.u_lambda = self.u_lambda#*0.9
        # check new sequences for composition and nutrition from domain knowledge
        return new_seq

                
    def test(self, epoch):
        self.training_model.eval()
        with torch.no_grad():
            new_seq, avg_reward = self.iteration(epoch, self.test_dataloader, train=False)
        
        return avg_reward

    def composition_check_and_mask(self, new_sequence):
        batch_size, seq_len = new_sequence.shape
        token_mispos_bool = torch.zeros_like(new_sequence)
        str_new_sequence = [list(map(self.vocab.convert_ids_to_tokens, seq)) for seq in new_sequence.detach().clone().cpu().numpy().tolist()]
        batch_mispos_bool = []

        for idx, seq in enumerate(str_new_sequence):
            temp = torch.zeros((seq_len,))
            mis_pos, mis_pos_idx = new_mispos_score(seq, self.food_columns_dict, self.morning_comb, self.lunch_comb, self.dinner_comb)
            if mis_pos != 0:
                batch_mispos_bool.append(1)
            else:
                batch_mispos_bool.append(0)
            temp[mis_pos_idx] = 1
            token_mispos_bool[idx] = temp
        
        return torch.tensor(batch_mispos_bool).int().to(self.device), token_mispos_bool.bool().to(self.device), str_new_sequence

    #def get_embedding_norm(self, allergy_condition):
    #    graph_emb = torch.load(self.graph_emb_path + ', '.join(allergy_condition) + '.pt')
    #    graph_emb_norm = (graph_emb - graph_emb.min()) / (graph_emb.max() - graph_emb.min())
    #    
    #    new_emb = torch.zeros((self.vocab.vocab_size, self.vocab.vocab_size))
    #    new_emb[:3, 3:] = 1.0
    #    new_emb[3:, 3:] = graph_emb_norm
    #    diag_idx = torch.arange(new_emb.size(0))
    #    new_emb[diag_idx, diag_idx] = 0
    #    
    #    return new_emb

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        
        avg_total_loss = 0.0
        avg_ult_loss = 0.0
        avg_ppo_loss = 0.0 
        avg_kl_loss = 0.0
        
        avg_reward = 0.0
        new_sequence_list = []
        
        for i, data in enumerate(data_loader):
            
            data = {key: value.to(self.device) for key, value in data.items()}
            
            encoder_input = data["encoder_input"] # encoder_input : [batch_size, seq_len]
            decoder_target = data["decoder_target"]
            attribute_mask = data["changed_attribute"] # attribute_mask: [batch_size, seq_len, num_attribute]
            original_attributes = data["original_attribute"]
            target_condition = data["target_condition"]
            mask_label = data["encoder_label"] # problem_multihot : [batch_size, seq_len] <- batch 별로 문제가 되는 sequence 위치에 1처리
            allergy_token_map = data["allergy_token_map"] # [batch_size, vocab_size]
            allergy_info_int = data["allergy_info"] # [batch_size] 
            
            #allergy_info_str = [data_loader.get_reverse_allergy_labels(allergy) for allergy in allergy_info_int.tolist()]
            
            #batch_graph_embed = torch.stack([self.graph_embeddings[idx.item()] for idx in allergy_info_int]).to(self.device)
            
            batch_size, seq_len = encoder_input.shape
            
            src_mask = self.training_model.make_src_mask(encoder_input)
            encoder_output = self.training_model.encoder(encoder_input, src_mask, attribute_mask)
            
            # 1. Autoregressive decoding
            if epoch == 0:
                sampled_diet = self.decoding_process(encoder_input, encoder_output,
                                                                  decoder_target, self.decoding, 
                                                                  allergy_token_map, old = True)
            else:
                sampled_diet = self.decoding_process(encoder_input, encoder_output,
                                                                  decoder_target, self.decoding, 
                                                                  allergy_token_map, old = False)
                #sampled_diet <s> token 포함
                
            # 2. 모델 확률 분포 (policy(y|x) 구하기)
            new_probs = self.training_model.forward(encoder_input, sampled_diet, attribute_mask)
            new_probs_entropy = self.logits_to_entropy(torch.exp(new_probs))
            mean_entropy = new_probs_entropy.mean(dim = -1)
            mean_entropy = mean_entropy.mean(dim=0)
            
            with torch.no_grad():
                old_probs = self.old_model.forward(encoder_input, sampled_diet, attribute_mask)
                
            kl = self.kl_loss(old_probs, new_probs)#, dim=-1
            #[batch, seq_len]
            
            reward = self.new_reward_function(sampled_diet[:, 1:], allergy_token_map) #<s> token제거
            avg_reward += reward.mean().item()
            
            #print(kl)
            
            ppo_loss = -1.*reward + self.kl_coef*kl - self.epsilon*mean_entropy
            loss = torch.mean(ppo_loss)
            
            #print(new_probs_entropy)
            
            avg_ppo_loss += torch.mean(ppo_loss).item()
            
            avg_kl_loss += kl.item()
            
            # Unlikelihood loss
            masked_lprobs = new_probs.reshape(-1, new_probs.size(2)).gather(1, sampled_diet[:, 1:].reshape(-1,1).to(self.device))
            #mask_label = mask_label.view(-1,1) #[batch*seq_len]
            #problem_idx = mask_label.nonzero(as_tuple=True)[0]
            
            pos_mask, neg_mask = self.check_and_mask(sampled_diet[:, 1:], target_condition, original_attributes) #
            batch_mispos_bool, token_mispos_bool, str_new_sequence = self.composition_check_and_mask(sampled_diet[:, 1:])
            token_mispos_bool = token_mispos_bool.view(-1,1)

            if self.u_lambda != 0:
                '''composition, allergy를 만족하지 않는 생성 token의 확률 낮추기 '''
                # -- Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
                
                one_minus_probs = torch.clamp((1.0 - masked_lprobs.exp()), min=1e-8)
                ult_mask =  neg_mask | token_mispos_bool
                #ult_mask = mask_label*ult_mask
                ult_loss = -torch.log(one_minus_probs)*(ult_mask)
                ult_loss = ult_loss.sum()
                
                ult_loss = self.u_lambda*ult_loss 
                loss += ult_loss
                avg_ult_loss += ult_loss.item()
                
            batch_correct = (~((neg_mask | token_mispos_bool).view(batch_size,seq_len).sum(dim=1) >0)).int() # 만족하는 sequence True
            batch_correct_idx = batch_correct.nonzero(as_tuple=True)[0]
            
            correct_str_new_sequence = [str_new_sequence[idx] for idx in batch_correct_idx.clone().cpu().numpy().tolist()]
            #correct_problem_multihot = target_condition[batch_correct_idx]
            #correct_problem_multihot = correct_problem_multihot.view(-1,1) # [batch_size*seq_len, 1]
            
            pass_idx = self.check_sequence_by_domain_knowledge(correct_str_new_sequence)
            pass_masking = torch.zeros((batch_size, seq_len))
            all_pass_idx = batch_correct_idx[pass_idx]
            pass_masking[all_pass_idx] = 1
            new_sequence_list += correct_str_new_sequence
            
            if train  : # & (epoch < 30)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            avg_total_loss += loss.item()
            
            
        if (epoch + 1 ) % 5 == 0:
            self.old_model.load_state_dict(self.training_model.state_dict())
            print("old model is updated !")

                
        avg_total_loss /= len(data_loader)
        avg_ult_loss /= len(data_loader)
        avg_ppo_loss /= len(data_loader)
        avg_reward /= len(data_loader)
            
        avg_kl_loss /= len(data_loader)

        print("Step%d_%s, avg_total_loss=" % (epoch, str_code), round(avg_total_loss,4), "avg_reward= {: .4f}".format(avg_reward), "PPO_loss={: .3f}".format(avg_ppo_loss), "utl_loss= {: .4f}".format(avg_ult_loss),"kl_loss={: .3f}".format(avg_kl_loss))
        return new_sequence_list, avg_reward
            
            
    def logits_to_entropy(self, logits):
        distribution = torch.distributions.Categorical(logits = logits)
        return distribution.entropy()
        
    def reward_function(self, sequence, allergy_token_map):
        
        # Reward 1: Non_Mispositioning (가중치 부여)
        # Reward 2: Beta Score
        # Reward 3: Non_Duplicates (Binary)
        # Reward 4: Non_Allergy rate (가중치 부여)
        
        str_sequence = [list(map(self.vocab.convert_ids_to_tokens, seq)) for seq in sequence.detach().clone().cpu().numpy().tolist()]
        
        mispos_reward = []
        duplicated_reward = []
        
        for _, seq in enumerate(str_sequence):
            mispos_score, _ = new_mispos_score(seq, self.food_columns_dict, self.morning_comb, self.lunch_comb, self.dinner_comb)
            
            mispos_score = (13 - mispos_score)/13
            mispos_reward.append(mispos_score)

            #beta_score = calculate_beta_score(seq, self.train_data.incidence_matrix, self.vocab)/13
            #beta_score_reward += beta_score
            
            duplicate = 1 - check_duplicates(seq)
            duplicated_reward.append(duplicate)
            
        allergy_numbers = allergy_token_map[torch.arange(sequence.size(0)).unsqueeze(1), sequence]
        nonallergy_reward = (sequence.size(1)) - allergy_numbers.sum(dim = 1)

        #print(nonallergy_reward)
            
        mispos_reward = mispos_reward
        nonallergy_reward = nonallergy_reward/(sequence.size(1))
        
        total_reward = torch.tensor(mispos_reward).to(self.device)*0.4 + nonallergy_reward*0.4 + torch.tensor(duplicated_reward).to(self.device)*0.2
        
        return total_reward
    
    
    def new_reward_function(self, sequence, allergy_token_map):
        
        # Reward 1: Non_Mispositioning (가중치 부여)
        # Reward 2: Non_Duplicates (Binary)
        # Reward 3: Non_Allergy rate (가중치 부여)
        # 모두 통과해야지 Reward를 부여 받음
        # 하나라도 부족하면, Reward를 받지 못함
        
        str_sequence = [list(map(self.vocab.convert_ids_to_tokens, seq)) for seq in sequence.detach().clone().cpu().numpy().tolist()]
        
        mispos_reward = [] #batch_size
        duplicated_reward = [] #batch_size
        
        for _, seq in enumerate(str_sequence):
            mispos_score, _ = new_mispos_score(seq, self.food_columns_dict, self.morning_comb, self.lunch_comb, self.dinner_comb)
            
            #mispos_score = mispos_score
            mispos_reward.append(mispos_score)

            #beta_score = calculate_beta_score(seq, self.train_data.incidence_matrix, self.vocab)/13
            #beta_score_reward += beta_score
            
            duplicate = check_duplicates(seq)
            duplicated_reward.append(duplicate)
            
        allergy_numbers = allergy_token_map[torch.arange(sequence.size(0)).unsqueeze(1), sequence]
        allergy_reward = allergy_numbers.sum(dim = 1) #batch_size
        
        #print(nonallergy_reward)
            
        #mispos_reward = mispos_reward
        
        total_reward = torch.tensor(mispos_reward).to(self.device) + allergy_reward+ torch.tensor(duplicated_reward).to(self.device)
        
        total_reward = (total_reward == 0).float()

        return total_reward
        
            
    def save_data(self, new_sequence_list, file_path): # masked_loc_list
        # dataset update
        num_new_seq = len(new_sequence_list)
        #filtered_sequence_list = self.check_sequence_by_domain_knowledge(new_sequence_list) # masked_loc_list
        
        filtered_sequence_list = remove_duplicates_2d_list(new_sequence_list)
        num_filtered_seq = len(filtered_sequence_list)
        print("num of added sample : {}/{}".format(num_filtered_seq, num_new_seq))
        
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
      
    def apply_repetition_penalty(self, logits, generated_tokens, penalty):
        
        for batch_idx, tokens in enumerate(generated_tokens):
            for token in tokens: 
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] /= penalty
                    
                else:
                    logits[batch_idx, token] *= penalty
        
        return logits 
            
    def decoding_process(self, encoder_input, encoder_output, decoder_target,
                     decoding, allergy_token_map, max_len=13, old = True):
        
        if old:
            model = self.old_model
        else:
            model = self.training_model
        
        model.eval()
        with torch.no_grad():
            
            #trg_mask = model.make_trg_mask(decoder_target)
            #cross_mask = model.make_cross_attn_mask(decoder_target, encoder_input)
            generated = decoder_target
            
            #sequence_probs = []
            for _ in range(max_len):
                
                trg_mask = model.make_trg_mask(generated)
                cross_mask = model.make_cross_attn_mask(generated, encoder_input)
                
                output = model.decoder(generated, trg_mask, encoder_output, cross_mask)
                output = model.mask_lm(output)
                
                next_token_logits = output[:, -1, :]
                # Repetition penalty
                #masked_logits = self.apply_repetition_penalty(next_token_logits, generated, 1.4)
                next_token = decoding(next_token_logits).unsqueeze(-1)
                
                
                #Graph
                #next_token_logits = torch.exp(next_token_logits)
                #next_token = decoding(next_token_logits).unsqueeze(-1)
                
                #new_probs = batch_graph_embed[torch.arange(encoder_input.size(0)), next_token.squeeze(1)] * next_token_logits
                #graph_embed_next_token = decoding(new_probs).unsqueeze(-1)
                
                #allergy_tokens_idx = np.where(allergy_token_map.detach().cpu()[torch.arange(encoder_input.size(0)),
                #                                                            next_token.squeeze(1).detach().cpu()] == 1)

                #next_token[allergy_tokens_idx] = graph_embed_next_token[allergy_tokens_idx]

                
                generated = torch.cat((generated, next_token), dim = 1)
                #sequence_probs.append(next_token_logits)
                if len(generated) == max_len+1:
                    break
        return generated#, torch.stack(sequence_probs, dim=0).transpose(0,1)

    def check_sequence_by_domain_knowledge(self, sequences):
        '''
        sequece : 2d list of sequence (string) [num_sequences, seq_len]
        '''
        #passed_sequences = []
        passed_idx = []
        #span_label = [1]*3 + [2]*2 + [3]*5 + [4]*2 + [5]*5
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
    
    
    def save(self, epoch, file_path="output/bart_trained.model/"):
        """
        Saving the current BART model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "model.ep%d" % epoch
        torch.save(self.training_model.cpu(), output_path)
        self.training_model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def best_save(self, epoch, file_path="output/bart_trained.model/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "model.best"
        torch.save(self.training_model.cpu(), output_path)
        self.training_model.to(self.device)
        print("[BEST]EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def check_and_mask(self, pred_tokens, target_condition, original_attribute):
        
        # pred_tokens : [batch_size, seq_len]
        # condition_multihot : [batch_size, num_attribute]
        # condition_keys : [batch_size, 1]

        #batch_indices = [torch.nonzero(condition_multihot[i], as_tuple=True)[0].tolist() for i in range(condition_multihot.shape[0])]
        batch_size, seq_len = pred_tokens.shape
        pred_tokens = pred_tokens.reshape(-1, 1) # [batch_size*seq_len, 1]
        pred_token_attributes = [self.VA_dictionary[token] for token in pred_tokens.squeeze(-1).cpu().tolist()]
        pred_token_attributes = torch.from_numpy(self.mlb.fit_transform(pred_token_attributes)).to(self.device) # [batch_size*seq_len, num_attribute]
        pred_token_attributes = pred_token_attributes.reshape(-1, pred_token_attributes.shape[-1]) # [batch_size*seq_len, num_attribute]

        # condition multi-hot  : [batch_size, num_attribute] -> 1개 시퀀스의 condition *batch

        target_condition = target_condition.repeat_interleave(seq_len, dim=0) #seq길이 만큼 target_condition반복
        #[batch_size*seq_len, num_attribute]
        
        token_level_checker = torch.any(pred_token_attributes*target_condition, dim = 1) #문제가 있으면 True, 없으면 False
        token_level_checker = token_level_checker.view(batch_size, seq_len)
        
        neg_mask = token_level_checker.reshape(-1, 1) #[batch*seq_len, 1]
        
        original_attribute = original_attribute.view(-1, original_attribute.size(-1))
        
        must = self.must_have_attribute(original_attribute, 0.5) #원래 재료의 attribute 50% 1
        
        pos_mask = torch.any(~pred_token_attributes.bool()& must, dim=1).int().view(-1, 1)
        #[batch*seq_len, 1] -> 있어야할 재료가 없으면 1 
        
        return pos_mask, neg_mask

    def must_have_attribute(self, original_attribute, rate):
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
        
        return original_por_attribute

