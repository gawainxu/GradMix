#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 22:30:17 2024

@author: zhi
"""

import torch
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import torch.nn as nn
from networks.resnet_big import model_dict, resnet18


class gradient_cache():
     
     """
     A reimplementation of gradient_cache in https://github.com/luyug/GradCache/
     """
     
     def __init__(self, model, splits, fp16, loss_fcn, grad_scalar=None, optimizer=None, if_normal=False, lam=1):
         
         """
         model: the model that to be trained
         inputs: list of inputs from two views, [tensor, tensor]
         splits: number of splits for one batch of data
         fp16: if use float16
         loss_fcn: loss function
         """
         
         self.model = model
         self.splits = splits
         self.loss_fcn = loss_fcn
         self.fp16 = fp16
         self.grad_scalar = grad_scalar
         self.if_normal = if_normal
         self.optimizer = optimizer
         self.reps_norm = []
         self.lam = lam
         
     def __call__(self, model_inputs, model_inputs_mix=None):
        
        return self.cache_step(model_inputs, model_inputs_mix)
    
    
     def split_inputs(self, inputs, splits):
         
         """
         inputs are converted to view images [x1, x2] [tensor, tensor]
         """
         bsz = int(inputs.shape[0] / 2)
         inputs = torch.split(inputs, [bsz, bsz], dim=0)
         inputs = [t.split(splits, dim=0) for t in inputs]
         # reorganize the data that each element in splits_inputs is list of two view of splited data, 
         #[[split_x1_1, split_x2_1], [split_x1_2, split_x2_2]]
         inputs = [list(s) for s in zip(*inputs)] 
         
         return inputs
     
        
     def model_call(self, model, x):
         
         return  model(x)
         
        
     def forward_no_grad(self, model_inputs):
         
         # read features without requiring gradient for one full batch
         # model_inputs is concatenated of two views
         
         with torch.no_grad():
              reps = self.model_call(self.model, model_inputs)
         #reps = self.model_call(model, model_inputs)                 # !!!!!!!
                
         return reps
     
        
     def compute_loss(self, reps, mixed_reps=None):
         
         loss = self.loss_fcn(features=reps, features_positive=mixed_reps)  # TODO check the format of loss_fcn
         
         return loss
     
        
     def build_cache(self, all_reps, mix_reps=None):
         
         """
         compute and store the gradients of the loss_fun over the representations
         """
         
         bsz = int(all_reps.shape[0] / 2)
         all_reps1, all_reps2 = torch.split(all_reps, [bsz, bsz], dim=0)
         #print(all_reps1[0], all_reps2[0])
         all_reps = torch.cat([all_reps1.unsqueeze(1), all_reps2.unsqueeze(1)], dim=1)
         #all_reps = all_reps.detach().requires_grad_()                             #!!!!
         #all_reps = all_reps.requires_grad_()         
         all_reps.requires_grad_().retain_grad()
         
         if mix_reps is None:
             loss = self.compute_loss(all_reps)
         else:
             mix_reps1, mix_reps2 = torch.split(mix_reps, [bsz, bsz], dim=0)
             mix_reps = torch.cat([mix_reps1.unsqueeze(1), mix_reps2.unsqueeze(1)], dim=1)
             mix_reps.requires_grad_().retain_grad()
             loss1 = self.compute_loss(all_reps)
             loss2 = self.compute_loss(reps=all_reps, mixed_reps=mix_reps)
             loss = loss1 + self.lam * loss2
              
         if self.fp16:
            self.grad_scalar.scale(loss).backward()
         else:
            loss.backward()
         
         if mix_reps is None:
             cache = all_reps.grad   # [bsz, 2, f] !!!!!  torch.ones_like(all_reps)  # 
             #print("feature gradient norm seperate", torch.norm(cache))
             return cache, loss
         else:
             cache = all_reps.grad
             cache_mix = mix_reps.grad
             #print("feature gradient norm seperate", torch.norm(cache), torch.norm(cache_mix))
             return cache, cache_mix, loss
     
        
     def forward_backward(self, model_inputs, reps_gradient_cache, model_inputs_mix=None, reps_grad_mix=None):
         
         for idx, (one_split_inputs, one_split_rep_grad) in enumerate(zip(model_inputs, reps_gradient_cache)):
             
             if model_inputs_mix is None:
                 one_split_inputs = torch.cat([one_split_inputs[0], one_split_inputs[1]], dim=0)  
                 one_split_reps = self.model_call(self.model, one_split_inputs)
                 one_split_reps.requires_grad_()
                 bsz = int(one_split_reps.shape[0] / 2)
                 one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
                 one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)   
                 surrogate = torch.sum(one_split_reps.flatten() * one_split_rep_grad.flatten())
             
             else:
                 one_split_inputs = torch.cat([one_split_inputs[0], one_split_inputs[1]], dim=0)  
                 one_split_reps = self.model_call(self.model, one_split_inputs)
                 bsz = int(one_split_reps.shape[0] / 2)
                 one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
                 one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)
                 one_split_reps.requires_grad_()
                
                 one_split_reps_grad_mix = reps_grad_mix[idx]
                 one_split_inputs_mix = model_inputs_mix[idx]
                 one_split_inputs_mix = torch.cat([one_split_inputs_mix[0], one_split_inputs_mix[1]], dim=0)
                 one_split_reps_mix = self.model_call(self.model, one_split_inputs_mix)         
                 one_split_reps_mix1, one_split_reps_mix2 = torch.split(one_split_reps_mix, [bsz, bsz], dim=0)
                 one_split_reps_mix = torch.cat([one_split_reps_mix1.unsqueeze(1), one_split_reps_mix2.unsqueeze(1)], dim=1)
                 one_split_reps_mix.requires_grad_()
                    
                 surrogate = torch.sum(one_split_reps_mix.flatten() * one_split_reps_grad_mix.flatten()) + torch.sum(one_split_reps.flatten() * one_split_rep_grad.flatten())
              
             """
             bsz = int(one_split_reps.shape[0] / 2)
             one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
             one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)
             one_split_reps.retain_grad()
             loss = self.compute_loss(one_split_reps)
             loss.backward()
             
             grad_norm = torch.norm(one_split_reps.grad)
             print("feature gradient norm full", grad_norm)
             self.reps_norm.append(grad_norm)
             """
                          
             surrogate.backward()
             norms = 0
             for p in self.model.parameters():
                 #print(p.grad.norm())
                 norms += p.grad.norm()
             print("model gradients norms", norms)
             
         return
     
        
     def seperated_oneshot(self, model_inputs):
         
         for one_split_inputs in model_inputs:
             one_split_inputs = torch.cat([one_split_inputs[0], one_split_inputs[1]], dim=0)
             with torch.no_grad():
                 one_split_reps = self.model_call(self.model, one_split_inputs)
              
             bsz = int(one_split_reps.shape[0] / 2)
             one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
             one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)
             one_split_reps.requires_grad_().retain_grad()
             loss = self.compute_loss(one_split_reps)
             loss.backward()
             
             reps_grad = one_split_reps.grad
             grad_norm = torch.norm(reps_grad)
             print("feature gradient norm full", grad_norm)
             
             self.optimizer.zero_grad()         #!!!!!!
             
             one_split_reps = self.model_call(self.model, one_split_inputs)
             one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
             one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)
             
             reps_grad.requires_grad_()
             surrogate = torch.sum(reps_grad.flatten() * one_split_reps.flatten())
             surrogate.backward()
             
             """
             #print(torch.norm(one_split_rep_grad))
             one_split_rep_grad = one_split_rep_grad.view(512, 128)
             surrogate = torch.sum(one_split_reps * one_split_rep_grad) #torch.dot(one_split_reps.flatten(), one_split_rep_grad.flatten())   # torch.dot(one_split_reps.flatten(), one_split_reps.flatten()) #
             surrogate.backward()
             """
             
         return loss
    

     def cache_step(self, model_inputs, model_inputs_mix=None):
         
         # read features for all data without gradients
         all_reps = self.forward_no_grad(model_inputs)
         if model_inputs_mix is not None:
             mix_reps = self.forward_no_grad(model_inputs_mix)
             
         # build cache
         if model_inputs_mix is not None:
             reps_grad_ori, reps_grad_mix, loss = self.build_cache(all_reps, mix_reps)
             reps_grad_ori = reps_grad_ori.split(self.splits, dim=0)
             reps_grad_mix = reps_grad_mix.split(self.splits, dim=0)
         else:
            reps_grad_ori, loss = self.build_cache(all_reps)
            # split again for pairing with splited gradients
            reps_grad_ori = reps_grad_ori.split(self.splits, dim=0)           #!!!!!!, (bsz, 2, f)
            reps_grad_mix = None

         # split data
         model_inputs = self.split_inputs(model_inputs, self.splits)
         if model_inputs_mix is not None:
             model_inputs_mix = self.split_inputs(model_inputs_mix, self.splits)
         else:
             model_inputs_mix = None
 
         # sub-batch gradient accumulation
         #self.optimizer.zero_grad()
         self.forward_backward(model_inputs, reps_grad_ori, model_inputs_mix=model_inputs_mix, reps_grad_mix=reps_grad_mix)           #!!!!!
         
         
         """
         # split data
         model_inputs = self.split_inputs(model_inputs, self.splits)
         loss = self.seperated_oneshot(model_inputs)
         """
         
         return loss
         
         
    
class gradient_cache_activations():
    
    def __init__(self, model, splits, loss_fcn, opt):
        
        self.model = model
        self.splits = splits
        self.loss_fcn = loss_fcn
        self.opt = opt
        self.hooks = []
        self.activations = []
        self.gradients = []
        
        if torch.cuda.device_count() > 1:
            if opt.method =="MoCo":
                self.encoder = self.model.encoder_q.module
            else:
                self.encoder = self.model.encoder.module
        else:
            if opt.method =="MoCo":
                self.encoder = self.model.encoder_q
            else:
                self.encoder = self.model.encoder
                
    
    def compute_loss(self, reps, mixed_reps=None):
          
        loss = self.loss_fcn(features=reps, features_positive=mixed_reps)  # TODO check the format of loss_fcn
        return loss
      
        
    def model_call(self, model, x):
            
        return  model(x)
    
        
    def __call__(self, model_inputs):
       
       return self.cache_step(model_inputs)
   
    
    def forward_no_grad(self, model_inputs):
         
         # read features without requiring gradient for one full batch
         # model_inputs is concatenated of two views
         
         with torch.no_grad():
              reps = self.model_call(self.model, model_inputs)
                
         return reps
   
    def split_inputs(self, inputs, splits):
        
        """
        inputs are converted to view images [x1, x2] [tensor, tensor]
        """
        bsz = int(inputs.shape[0] / 2)
        inputs = torch.split(inputs, [bsz, bsz], dim=0)
        inputs = [t.split(splits, dim=0) for t in inputs]
        inputs = [list(s) for s in zip(*inputs)] 
        
        return inputs
    
    
    def build_cache(self, all_reps):
        
        """
        compute and store the gradients of the loss_fun over the representations
        """
        
        bsz = int(all_reps.shape[0] / 2)
        all_reps1, all_reps2 = torch.split(all_reps, [bsz, bsz], dim=0)
        all_reps = torch.cat([all_reps1.unsqueeze(1), all_reps2.unsqueeze(1)], dim=1)    
        all_reps.requires_grad_().retain_grad()

        loss = self.compute_loss(all_reps)
        loss.backward()
        
        cache = all_reps.grad
        print("feature gradient norm seperate", torch.norm(cache))
        return cache, loss
    
    
    def forward_backward(self, model_inputs, reps_gradient_cache):
        
        for idx, (one_split_inputs, one_split_rep_grad) in enumerate(zip(model_inputs, reps_gradient_cache)):
            one_split_inputs = torch.cat([one_split_inputs[0], one_split_inputs[1]], dim=0)
            one_split_reps = self.model_call(self.model, one_split_inputs)
            
            #print(torch.norm(one_split_rep_grad))
            one_split_reps.requires_grad_()
            
            bsz = int(one_split_reps.shape[0] / 2)
            one_split_reps1, one_split_reps2 = torch.split(one_split_reps, [bsz, bsz], dim=0)
            one_split_reps = torch.cat([one_split_reps1.unsqueeze(1), one_split_reps2.unsqueeze(1)], dim=1)   
            surrogate = torch.sum(one_split_reps.flatten() * one_split_rep_grad.flatten())
            
            surrogate.backward()
            
        return
    
    
    def save_activation_hook(self, _, input, output):
        # TODO check how forward hook is saved 

        print("forward hook")
        self.activations.append(output.detach())


    def save_backward_hook(self, _, input, output):
        
        print("backward hook")
        self.gradients.append(output[0].detach())
        
    
    def cache_step(self, model_inputs):
        
        all_reps = self.forward_no_grad(model_inputs)
        reps_gradient_cache, loss = self.build_cache(all_reps)
        reps_gradient_cache = reps_gradient_cache.split(self.splits, dim=0) 
        
        model_inputs = self.split_inputs(model_inputs, self.splits)
        self.hooks.append(self.encoder.layer4[-1].register_forward_hook(hook=self.save_activation_hook))
        self.hooks.append(self.encoder.layer4[-1].register_full_backward_hook(hook=self.save_backward_hook))
        self.forward_backward(model_inputs, reps_gradient_cache)        
        
        for h in self.hooks:
            h.remove()
            
        self.activations = torch.cat(self.activations, dim=0)
        self.gradients = torch.cat(self.gradients, dim=0)
            
        return self.activations, self.gradients
        
    
"""
         # read features without grad
         for x in splited_inputs:
             one_split_reps = self.forward_no_grad(self.model, x)
             all_reps.append(one_split_reps)                         # splited, [splits, 2, fd]
             
         # TODO concatenate all sub-batch representations 
         all_reps1 = [ap[0] for ap in all_reps]
         all_reps2 = [ap[1] for ap in all_reps]
         all_reps1 = torch.cat(all_reps1, dim=0)
         all_reps2 = torch.cat(all_reps2, dim=0)
         all_reps1.requires_grad_()
         all_reps2.requires_grad_()
         all_reps = [all_reps1, all_reps2]    # .detach().requires_grad_()
"""

"""
for p in self.model.parameters():
    print(p.grad.norm())
"""


class MoCoResNet(nn.Module):
    
    def __init__(self, opt, name="resnet", head="linear", feat_dim=128, in_channels=3):
        """
        K: size of the bank buffer
        """
        super(MoCoResNet, self).__init__()
        self.queue_size = opt.K
        self.momentum = opt.momentum_moco
        self.temp = opt.temp
        self.temp_cache = []
        
        model_fun, dim_in = model_dict[name]
        self.encoder_q = model_fun(in_channels=in_channels)
        self.encoder_k = model_fun(in_channels=in_channels)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)        # initialize
            param_k.requires_grad = False           # no update by gradient

        if head == "linear":
            self.head_q = nn.Linear(dim_in, feat_dim)
            self.head_k = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head_q = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.head_k = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)        # initialize
            param_k.requires_grad = False           # no update by gradient

        # create the queue to store negative samples
        self.register_buffer("queue", torch.randn(self.queue_size, feat_dim))   # TODO attention the dimension here
        #self.queue = nn.functional.normalize(self.queue, dim=1)

        # create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        """
        update the key_encoder parameters through the momemtum updata
        
        key_parameters = momemtum * key_parameters + (1 - momentum) * query_parameters
        """

        # for each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)
        
        for p_q, p_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):

        # generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()
        reverse_idxs = torch.zeros(batch_size).long().cuda()
        value = torch.arange(batch_size).long().cuda()
        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, feat_k):

        batch_size = feat_k.size(0)
        ptr = int(self.queue_ptr)
        print(batch_size, feat_k.size())

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr+batch_size, :] = feat_k      # TODO attention the dimensions here !!!!

        # move pointer alone the end of current batch
        ptr = (ptr + batch_size) % self.queue_size

        # store queue pointer as register_buffer
        self.queue_ptr[0] = ptr


    def InfoNCE_logits(self, f_q, f_k):

        """
        compute the similarity logits between positive samples and
        positive to all negative in the memory
        """

        f_k = f_k.detach()

        # get queue from register_buffer
        f_mem = self.queue.clone().detach()

        # normalize the feature representation
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)

        # compute sim between positive views
        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1), f_k.view(f_k.size(0), -1, 1)).squeeze(-1)     # bmm((bsz, 1, dim), (bsz, dim, 1)) = (bsz, 1, 1), sim between corresponding tensors => (bsz, 1)

        # compute sim between positive and all negative in memory
        neg = torch.mm(f_q, f_mem.transpose(1, 0))         # mm((bsz, dim), (dim, bsz)) = (bsz, bsz)

        logits = torch.cat((pos, neg), dim=1)              # (bsz, bsz+1), the first is the sim between f_q and f_k, the rest bsz are the sims between f_q and f_mem in the queue
        logits /= self.temp

        # create labels, first logit is posive and the rest are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels


    def forward(self, x_q, x_k=None, mode="simclr", update_cache=False):

        batch_size = x_q.size(0)
        #print("temp cache", len(self.temp_cache))

        # feature of the query view from the query enoder
        feat_q = self.head_q(self.encoder_q(x_q))

        if mode == "moco":
            # get shuffled and reversed indexes for the current minibatch
            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)
            with torch.no_grad():
                # update the key encoder
                self.momentum_update()                    
                # shuffle minibatch
                x_k = x_k[shuffled_idxs]
                # feature representations of the shuffled key view from the key encoder
                feat_k = self.head_k(self.encoder_k(x_k))
                # reverse the shuffled samples to original position
                feat_k = feat_k[reverse_idxs]
                
            # compute the logits for the InfoNCE contrastive loss
            logit, labels = self.InfoNCE_logits(feat_q, feat_k)
            # updata the queue/memory with the curent key_encoder minibatch
            if update_cache:
                self.temp_cache.append(feat_k)
                self.update_queue(torch.cat(self.temp_cache, dim=0))
                self.temp_cache = []
            else:
                self.temp_cache.append(feat_k)

            return logit, labels

        elif mode == "simclr":
            assert x_k == None
            feat_q = nn.functional.normalize(feat_q, dim=1)
            
            return feat_q