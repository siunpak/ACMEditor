import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model.Encoder import AGGBART_Encoder
from model.Decoder import AGGBART_Decoder
from model.BART import AGGBART
from source.f_dataloader_withpair import FinetuneDataset, FinetuneDataset_with_RP

from source.dataloader import Attribute, Vocab, DietDataset
from source.utils import create_vocab_attribute_dictionary, remove_duplicates_2d_list, save_to_csv
from source.metric import new_mispos_score, check_duplicates
from source.sampler import TemperatureSampler, NucleusSampler, GreedySampler, TopKSampler
import math
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.tensorboard import SummaryWriter
import tqdm
import copy
import pandas as pd
import random
import torch.optim.lr_scheduler as lr_scheduler


class AAGBARTFINETUNER_WITH_PAIR:

    def __init__(self, bart, 
                 vocab:Vocab, attribute:Attribute, train_data: FinetuneDataset_with_RP,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), 
                 with_cuda: bool = True, cuda_devices=None, 
                 u_lambda = 0.1, u_decode="argmax", u_degrade=False,
                 agm = True, use_attribute_space= False, sampling = "random", k = 5,
                 food_columns_dict = None, morning_comb=None, lunch_comb=None, 
                 dinner_comb = None):
        """
        :param bart: pretrained BART model
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
        import os
        os.environ["CUDA_VISIBLE_DEVICE"]="3"
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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

        self.use_attribute_space = use_attribute_space
        self.sampling = sampling
        self.k = k
        if use_attribute_space:
            self.vocab_attribute_reference = train_data.vocab_attribute_reference
            self.allergy_satified_vocab_dict = train_data.allergy_satified_vocab_dict
        #self.allergy_combination_dict = train_data.allergy_combination_dict
        
        self.food_columns_dict = food_columns_dict
        self.morning_comb = morning_comb
        self.lunch_comb = lunch_comb
        self.dinner_comb = dinner_comb

        self.mlb = MultiLabelBinarizer(classes=np.arange(self.attribute.attribute_size))

        self.model = bart.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        #if with_cuda and torch.cuda.device_count() > 1:
        #    print("Using %d GPUS for BART" % torch.cuda.device_count())
        #    self.model= nn.DataParallel(self.model, device_ids=[1,3]).cuda()

        #self.TransitionMatrix = TransitionMatrix(vocab.vocab_size)
        #self.replaybuffer = ReplayBuffer(capacity=1000000, seq_len=17, num_attributes=attribute.attribute_size)

        self.num_predictor = nn.Linear(512,1).to(self.device)
        self.p_cirterion = nn.BCELoss()

        # setting unlikelihood training
        #self.u_lambda= u_lambda
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=0.001)#0.001)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)

        self.writer = SummaryWriter()

        self.train_step = 0
        self.test_step = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    

    def train(self, epoch):
        print("num of corpus :", self.train_data.__len__())
        self.model.train()
        avg_correct, augmented_data, original_data, original_allergy = self.iteration_simple(epoch, self.train_dataloader)
        #avg_afc_ratio, augmented_data, original_data, original_allergy = self.iteration_simple(epoch, self.train_dataloader)
        if self.u_degrade=="True":
            self.u_lambda = self.u_lambda*0.9
        # check new sequences for composition and nutrition from domain knowledge
        return avg_correct, augmented_data, original_data, original_allergy

                
    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            avg_correct, _, _, _ = self.iteration_simple(epoch, self.test_dataloader, train=False)
            #avg_correct, _, _, _= self.iteration_simple(epoch, self.test_dataloader, train=False)
        
        return avg_correct

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

    def iteration_simple(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        avg_total_loss = 0.0
        avg_ult_loss = 0.0
        correct_count = 0
        correct_label = 0
        
        avg_pred_loss = 0.0
        n_element = 0
        label_elements = 0
        
        new_sequence_list = []
        original_sequence_list = []
        original_allergies = []

        for i, data in enumerate(data_loader):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            
            encoder_input = data["encoder_input"] # encoder_input : [batch_size, seq_len]
            decoder_target = data["decoder_target"]
            attribute_mask = data["changed_attribute"] # attribute_mask: [batch_size, seq_len, num_attribute]
            original_attributes = data["original_attribute"]
            target_condition = data["target_condition"]
            mask_label = data["encoder_label"] # problem_multihot : [batch_size, seq_len] <- batch 별로 문제가 되는 sequence 위치에 1처리
            allergy_info = data["allergy_info"]
            
            batch_size, seq_len = encoder_input.shape
            
            # 1. Conditioned attribute prediction
            #src_mask = self.model.make_src_mask(encoder_input)
            
            model_probs = self.model.forward(encoder_input, decoder_target[:, :-1], attribute_mask)
            pred_loss = self.criterion(model_probs.transpose(1,2), decoder_target[:, 1:])
            
            loss = pred_loss.sum()
            avg_pred_loss += pred_loss.item()
            
            generated_tokens = self.decoding(model_probs[:, :-1, :])
            # generated_tokens : [batch, seq_len], model_probs : [batch, seq_len, vocab_size]
            

            #예측된 token의 확률값 찾기 -> 예측된 거 + allergy_token_map에 해당하는 것 
            # -> allergy_token_map에 해당하는 확률값은 전부 unlikelihood training
            
            masked_lprobs = model_probs.reshape(-1, model_probs.size(2)).gather(1, generated_tokens.reshape(-1, 1).to(self.device))
            # [batch*seq_len, 1] # 마스킹 된 지점에 대한 unlikelihood loss
                
            # 문제가 되었던 지점 찾기  
            #mask_label = mask_label.view(-1,1) #[batch*seq_len]
            #problem_idx = mask_label.nonzero(as_tuple=True)[0]

            # allergy check & attribute check : 문제가 있으면 1, 없으면 0
            pos_mask, neg_mask = self.check_and_mask(generated_tokens, target_condition, original_attributes) #
            
            # neg_mask는 무조건 충족
            
            # composition check, composition에 문제가 있으면 1, 없으면 0. 배치 단위, token 단위 모두 뽑아야 함.
            batch_mispos_bool, token_mispos_bool, str_new_sequence = self.composition_check_and_mask(generated_tokens)

            token_mispos_bool = token_mispos_bool.view(-1,1)
            
            if self.u_lambda != 0: 
                '''composition, allergy를 만족하지 않는 생성 token의 확률 낮추기 '''
                # -- Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
                
                one_minus_probs = torch.clamp((1.0 - masked_lprobs.exp()), min=1e-8)
                ult_mask = neg_mask | token_mispos_bool
                #ult_mask = mask_label*ult_mask
                ult_loss = -torch.log(one_minus_probs)*(ult_mask)
                ult_loss = ult_loss.sum()
                
                ult_loss = self.u_lambda*ult_loss #/ len(ult_mask.nonzero(as_tuple=True)[0])
                loss += ult_loss
                avg_ult_loss += ult_loss.item()
                

            # allergy, composition, attribute 모두 만족하는 배치 내 seqence 찾기
            batch_correct = (~((neg_mask | token_mispos_bool).view(batch_size,seq_len).sum(dim=1) >0)).int() # 만족하는 sequence True
            batch_have_target = (~(target_condition.sum(dim=1) == 0)).int()
            
            batch_correct_idx = (batch_correct & batch_have_target).nonzero(as_tuple=True)[0]
            
            #batch_correct_idx = batch_correct.nonzero(as_tuple=True)[0]
            
            correct_str_new_sequence = [str_new_sequence[idx] for idx in batch_correct_idx.clone().cpu().numpy().tolist()]
            
            pass_idx = self.check_sequence_by_domain_knowledge(correct_str_new_sequence)
            pass_masking = torch.zeros((batch_size, seq_len))
            all_pass_idx = batch_correct_idx[pass_idx]
            pass_masking[all_pass_idx] = 1
            
            #data add
            correct_new_sequence = [correct_str_new_sequence[idx] for idx in pass_idx]
            
            original_sequence = encoder_input[all_pass_idx]
            original_sequence = [list(map(self.vocab.convert_ids_to_tokens, seq)) for seq in original_sequence.detach().clone().cpu().numpy().tolist()]
            
            original_allergy_info = allergy_info[all_pass_idx].detach().tolist()
            original_allergy_info = [self.train_data.get_reverse_allergy_labels(info) for info in original_allergy_info]
            
            original_sequence_list += original_sequence
            new_sequence_list += correct_new_sequence
            original_allergies += original_allergy_info

            # 3. backward and optimization only in train
            if train  : # & (epoch < 30)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            real = mask_label
            idx_real_nonzero = (real > 0)
            pred_mask = torch.mul(idx_real_nonzero, torch.argmax(model_probs[:, :-1, :].transpose(1, 2), dim=1).detach())
            pred_element = pred_mask[idx_real_nonzero]
            real_element = real[idx_real_nonzero]
            correct_count += (pred_element == real_element).sum().item()
            n_element += len(real_element)
            
            # allergy 예측
            #real_label_idx = np.where(decoder_target[:, 15:-1].detach().cpu().reshape(-1, 1) != 1)[0]
            #real_label = decoder_target[:, 15:-1].detach().cpu().reshape(-1, 1)[real_label_idx]
            #pred_label = self.decoding(model_probs[:, -6:-1, :]).detach().cpu().reshape(-1,1)[real_label_idx]
            
            #correct_label += (pred_label == real_label).sum().item()
            #label_elements += len(real_label)
            
            avg_total_loss += loss.item()
        #    mean_af_normal_ratio = len(batch_correct_idx)/batch_size
        #    avg_afc_ratio += mean_af_normal_ratio

            if train:
                step = self.train_step
            else:
                step = self.test_step
            #    
            #self.writer.add_scalar("FT/Loss/{}".format(str_code), mean_loss , step)
            #self.writer.add_scalar("FT/ult/{}".format(str_code), mean_ult_loss , step)
            #self.writer.add_scalar("FT/afc/{}".format(str_code), mean_af_normal_ratio , step)
            #self.writer.add_scalar("FT/mlt/{}".format(str_code), mean_mlt_loss , step)

            if train:
                self.train_step += 1
            else:
                self.test_step += 1
                
        self.scheduler.step()

    #    avg_afc_ratio /= len(data_loader)
        avg_total_loss /= len(data_loader)
        avg_ult_loss /= len(data_loader)
        #avg_mlt_loss /= len(data_loader)
        # avg_condi_loss /= len(data_loader)
    #    avg_pred_loss /= len(data_loader)
        avg_correct = correct_count*100 / n_element
        #avg_correct_label = correct_label*100 / label_elements
        
        print("EP%d_%s, avg_total_loss=" % (epoch, str_code), round(avg_total_loss,4), "utl_loss= {:.4f}".format(avg_ult_loss), "avg_correct={:.4f}".format(avg_correct))#, "avg_correct_label={:.4f}".format(avg_correct_lbel))
        return avg_correct, new_sequence_list, original_sequence_list, original_allergies
    
    def save_data(self, new_sequence_list, original_sequence, original_allergies, file_path): # masked_loc_list
        # dataset update
        combined_data = [a+b+[','.join(c)] for a,b,c in zip(new_sequence_list, original_sequence, original_allergies)]
        
        num_new_seq = len(combined_data)
        #filtered_sequence_list = self.check_sequence_by_domain_knowledge(new_sequence_list) # masked_loc_list
        
        filtered_sequence_list = remove_duplicates_2d_list(combined_data)
        num_filtered_seq = len(filtered_sequence_list)
        print("num of added sample_w/o_duplicates : {}/{}".format(num_filtered_seq, num_new_seq))
        
        if num_filtered_seq !=0:
            # save dataset
            header = list(range(0,26)) + ['allergy']
            
            save_to_csv(file_path, filtered_sequence_list, header)
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
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

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
