# 分别用两个transformer处理视频和文本，再拼接后用一个transformer的self-attention左融合

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict

faulthandler.enable()
import utils
from seq_scripts_GSL_bert_text_nsp_slt_german_bias_slr_gai import seq_train, seq_eval, seq_feature_generation  # seq_scripts_GSL_bert_text_nsp_slt_sen, seq_scripts_GSL_bert_text_nsp_slt_german_bias_slr
from pytorch_pretrained_bert.tokenization import BertTokenizer
from thop import profile
from torchvision.models import resnet50
from torchvision.models import inception_v3
# model=resnet50()
# input=torch.randn(1,3,224,224)
# flops,params=profile(model,inputs=(input,))
# input = torch.FloatTensor(280,3,180,180)
# model1=inception_v3()
# flops, params = profile(model1, inputs=(input,))
# print('flops:', flops)
# print('params:', params)
# print(frame_feat.size())
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}


        # new label
        ch_vocab_filename = self.arg.dict
        # ch_vocab_filename = "DatasetFile/dialogue_tju_QA_new/a_vocab.txt"
        with open(ch_vocab_filename, encoding='utf-8') as f:
            vocab = f.readlines()
            self.gloss_dict = {vocab[i].replace('\n', ''):i+1 for i in range(len(vocab))}

        # new label
        ch_vocab_text_filename = self.arg.dict_text
        with open(ch_vocab_text_filename, encoding='utf-8') as f:
            vocab = f.readlines()
            self.text_dict = {vocab[i].replace('\n', ''):i+1 for i in range(len(vocab))}
            self.text_dict[PAD_TOKEN] = 0
            # self.text_dict[BOS_TOKEN] = 1
            # self.text_dict[EOS_TOKEN] = 2

        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        # self.arg.model_args['num_classes_T'] = len(self.gloss_dict_T) + 1
        # self.arg.model_args['num_classes_E'] = len(self.gloss_dict_E) + 1
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,self.device,
                          epoch, self.recoder)
                if eval_model:
                    try:
                        dev_wer = 0
                        test_wer = 0
                        # dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                        #                 'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool, "QA")
                        dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                           'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))

                        # test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                        #                 'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool, "QA")
                        test_wer = seq_eval(self.arg, self.data_loader['test'], self.model, self.device,
                                            'test', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                        self.recoder.print_log("test WER: {:05.2f}%".format(test_wer))
                    except Exception as e:
                        print(e)
                        pass
                if save_model:
                    model_path = "{}dev_{:05.2f}test_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer,test_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            # **self.arg.bert_args,
            bert_arg = self.arg.bert_args,
            slt_arg = self.arg.slt_args,
            gloss_dict = self.gloss_dict,
            text_dict = self.text_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            drop_key = set()
            for w in self.arg.ignore_weights:
                for key in state_dict['model_state_dict'].keys():
                    if key.startswith(w):
                        drop_key.add(key)
            for key in drop_key:
                if state_dict['model_state_dict'].pop(key, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(key))
                else:
                    print('Can Not Remove Weights: {}.'.format(key))
        weights = self.modified_weights(state_dict['model_state_dict'], False)  ##gloss_clip_question
        # weights = self.modified_weights(state_dict['model_state_dict'],True)
        model.load_state_dict(weights, strict=False)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        # dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        dataset_list = zip(["train", "dev","test"], [True, False,False])

        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            tokenizer = BertTokenizer.from_pretrained(
                self.arg.bert_args['bert_model'],
                cache_dir=args.bert_args['output_dir'] + '/.pretrained_model_{}'.format(args.bert_args['global_rank']))
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict,text_dict = self.text_dict,tokenizer=tokenizer, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    # sparser = utils.get_parser()
    sparser = utils.parameters_TJU_QA_bert_sign_text_graph_slt.get_parser()

    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        # for k in default_arg.keys():
        #     if k not in key:
        #         print('WRONG ARG: {}'.format(k))
        #         assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
