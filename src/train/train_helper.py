import json
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ExponentialLR
from model.bert_bilstm_crf import BertBiLstmCrf, cal_bert_bilstm_crf_loss
from model.bilstm import BiLSTM, cal_bilstm_loss
from model.bilstm_crf import BiLSTM_CRF, cal_bilstm_crf_loss
from tools.get_ner_level_acc import precision
from tools.help import load_pickle_obj, sort_by_lengths, batch_sents_to_tensorized
from model.path import get_model_dir, get_tag2id_path, get_word2id_path

torch.manual_seed(1234)

class NerModel(object):
    def __init__(self, vocab_size, out_size, use_pretrained_w2v=False, model_type="bilstm-crf"):
        self.model_dir = get_model_dir()
        self.model_type_dir = os.path.join(self.model_dir, model_type)
        if not os.path.exists(self.model_type_dir):
            os.makedirs(self.model_type_dir)
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.out_size = out_size
        self.batch_size = 64
        self.lr = 0.01
        self.epoches = 25
        self.print_step = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用 : {self.device} ...")
        self.emb_size = 768
        self.hidden_size = 256
        self.dropout = 0.5
        self.use_pretrained_w2v = use_pretrained_w2v
        if self.model_type == "bilstm-crf":
            self.model = BiLSTM_CRF(self.vocab_size, self.emb_size, self.hidden_size, self.out_size, self.dropout, self.use_pretrained_w2v)
            self.loss_cal_fun = cal_bilstm_crf_loss
        elif self.model_type == "bilstm":
            self.model = BiLSTM(self.vocab_size, self.emb_size, self.hidden_size, self.out_size, self.dropout, self.use_pretrained_w2v)
            self.loss_cal_fun = cal_bilstm_loss
        elif self.model_type == "bert-bilstm-crf":
            self.model = BertBiLstmCrf(self.vocab_size, self.emb_size, self.hidden_size, self.out_size, self.dropout, self.use_pretrained_w2v)
            self.loss_cal_fun = cal_bert_bilstm_crf_loss

        self.model.to(self.device)   
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=0.005)
        self.scheduler = ExponentialLR(self.optimizer, gamma = 0.8)

        self.step = 0
        self.best_val_loss = 1e18
        if self.use_pretrained_w2v:
            model_name = f"{self.model_type}-pretrained.pt"
        else:
            model_name = f"{self.model_type}.pt"
        self.model_save_path = os.path.join(self.model_type_dir, model_name)
        
        # 添加用于绘制loss曲线的列表
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists, word2id, tag2id):
        train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        total_step = (len(train_word_lists)//self.batch_size + 1)

        epoch_iterator = trange(1, self.epoches + 1, desc="Epoch")
        for epoch in epoch_iterator:
            self.step = 0
            loss_sum = 0.
            epoch_loss = 0.
            batch_count = 0
            
            for idx in trange(0, len(train_word_lists), self.batch_size, desc="Iteration:"):
                batch_sents = train_word_lists[idx : idx + self.batch_size]
                batch_tags = train_tag_lists[idx : idx + self.batch_size]
                loss = self.train_step(batch_sents, batch_tags, word2id, tag2id)
                loss_sum += loss
                epoch_loss += loss
                batch_count += 1
                
                if self.step == total_step:
                    print("\nEpoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        loss_sum / self.print_step
                    ))
                    loss_sum = 0.
            
            # 记录每个epoch的平均训练loss
            avg_train_loss = epoch_loss / batch_count
            self.train_losses.append(avg_train_loss)
            print(f"Epoch {epoch} 平均训练 loss: {avg_train_loss:.4f}")
            
            # 验证
            val_loss = self.validate(epoch, dev_word_lists, dev_tag_lists, word2id, tag2id)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            if epoch > 10 and self.model_type != "bert-bilstm-crf":
                self.test(test_word_lists, test_tag_lists, word2id, tag2id)
            elif epoch > 15:
                self.test(test_word_lists, test_tag_lists, word2id, tag2id)
            
            # 每个epoch结束后更新loss曲线图
            self.plot_loss_curves()

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step+=1
        batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(batch_sents, word2id)
        labels_tensor, _ = batch_sents_to_tensorized(batch_tags, tag2id)

        batch_sents_tensor, labels_tensor = batch_sents_tensor.to(self.device), labels_tensor.to(self.device)
        scores = self.model(batch_sents_tensor, sents_lengths)

        self.model.zero_grad()
        loss = self.loss_cal_fun(scores, labels_tensor, tag2id)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    
    def validate(self, epoch, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_step = 0
            for idx in range(0, len(dev_word_lists), self.batch_size):

                val_step+=1
                batch_sents = dev_word_lists[idx : idx + self.batch_size]
                batch_tags = dev_tag_lists[idx : idx + self.batch_size]
                batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(batch_sents, word2id)
                labels_tensor, _ = batch_sents_to_tensorized(batch_tags, tag2id)
                batch_sents_tensor, labels_tensor = batch_sents_tensor.to(self.device), labels_tensor.to(self.device)
                scores = self.model(batch_sents_tensor, sents_lengths)

                loss = self.loss_cal_fun(scores, labels_tensor, tag2id).item()

                val_loss += loss
            
            avg_val_loss = val_loss / val_step if val_step > 0 else val_loss
            print(f"------epoch: {epoch}, val loss: {avg_val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                print(f"保存模型，path: {self.model_save_path}")
                torch.save(self.best_model.state_dict(), self.model_save_path)
                print(f"curren best val loss: {self.best_val_loss}")
            
            return avg_val_loss

    def plot_loss_curves(self):
        """绘制训练和验证loss曲线并保存"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='训练loss')
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='验证loss')
        
        plt.title(f'{self.model_type} 训练和验证loss曲线')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存loss曲线图
        curves_dir = os.path.join(self.model_type_dir, "curves")
        if not os.path.exists(curves_dir):
            os.makedirs(curves_dir)
            
        loss_curve_path = os.path.join(curves_dir, f"{self.model_type}_loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        
        # 同时保存loss数据，便于后续分析
        loss_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        loss_data_path = os.path.join(curves_dir, f"{self.model_type}_loss_data.json")
        with open(loss_data_path, 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=4)

    def test(self, test_word_lists, test_tag_lists, word2id, tag2id):
        test_word_lists, test_tag_lists, indices = sort_by_lengths(test_word_lists, test_tag_lists)
        batch_sents_tensor, sents_lengths = batch_sents_to_tensorized(test_word_lists, word2id)
        batch_sents_tensor = batch_sents_tensor.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.predict(batch_sents_tensor, sents_lengths, tag2id)
        pre_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.model_type.find("bilstm-crf") != -1:
                for j in range(sents_lengths[i] - 1):
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(sents_lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])  
            pre_tag_lists.append(tag_list)           
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pre_tag_lists = [pre_tag_lists[i] for i in indices]
        tag_lists = [test_tag_lists[i] for i in indices]

        total_precision, result_dic = precision(pre_tag_lists, tag_lists)
        print(f"实体级准确率为: {total_precision}")
        print(f"各实体对应的准确率为: {json.dumps(result_dic, ensure_ascii=False, indent=4)}")

    
    def predict(self, text):
        text_list = list(text)
        if self.model_type.find("bilstm-crf") != -1:
            text_list.append("<end>")
        text_list = [text_list]
        word2id_path = get_word2id_path()
        tag2id_path = get_tag2id_path()
        word2id = load_pickle_obj(word2id_path)
        tag2id = load_pickle_obj(tag2id_path)

        tensorized_sents, lengths = batch_sents_to_tensorized(text_list, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        model_path = self.model_save_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            batch_tagids = self.model.predict(tensorized_sents, lengths, tag2id)
        pre_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.model_type.find("bilstm-crf") != -1:
                for j in range(lengths[i] - 1):
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])  
            pre_tag_lists.append(tag_list)           
        return pre_tag_lists

    def load_and_test(self, test_word_lists, test_tag_lists, word2id, tag2id):
        """加载已保存模型并在测试集上评估"""
        if not os.path.exists(self.model_save_path):
            print(f"模型文件不存在: {self.model_save_path}")
            return
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.eval()
        print(f"加载模型: {self.model_save_path}")
        # 兼容 self.best_model
        self.best_model = self.model
        self.test(test_word_lists, test_tag_lists, word2id, tag2id)