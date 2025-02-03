import time
import torch.nn.functional as F

import math
import torch
from torch import cuda
from torch.autograd import gradcheck
from Linear_Attention import FASTMultiHeadAttention
import fastmax_cuda
import numpy as np


LA_ours = FASTMultiHeadAttention() # ours linear attention implementation
LA_torch = FASTMultiHeadAttention(False) # linear attention implemented using pytorch



# look here
b = 4 # batch
h = 16 # head
d = 64 # dimension per head (i.e. embedding_dimension/h)

# n changes from 10^strt to 10^endd. The number of test points are count
rep = 100
count = 4
strt = 3 # log scale
endd = 5 # log scale
# lengths = [1024,4096,8192]


dtype = torch.float32
print("bhd = ",b,",",h,",",d,",")

reg_attention_time = np.zeros(count)
our_LA_time = np.zeros(count)
torch_LA_time = np.zeros(count)
reg_attention_memory = np.zeros(count)
our_LA_memory = np.zeros(count)
torch_LA_memory = np.zeros(count)
device = torch.device(0)
mask = True


j = -1
print("Our LA Implementation")
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            # print(ii)
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'),requires_grad=True, dtype=dtype).contiguous()
            start_time = time.time()
            e = LA_ours(q,k,v,mask)
            print(e)
            cuda.synchronize()
            end_time = time.time()
            our_LA_time[j] += (end_time - start_time)/rep
        our_LA_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))


print("############################################")
print("Regular Attention")

j = -1
for i in np.logspace(strt, endd, count):
#   for i in lengths:
    try:
        j += 1
        print(int(i))
        if(i > 500000): continue
        for ii in range(rep):
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)

            start_time = time.time()
            e = F.scaled_dot_product_attention(q, k, v,is_causal=True)
            # e = torch.sum(F.scaled_dot_product_attention(q, k, v,is_causal=True)).backward(retain_graph=True)
            cuda.synchronize()
            end_time = time.time()
            reg_attention_time[j] += (end_time - start_time)/rep
        reg_attention_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))


print("############################################")

j = -1
print("Pytorch LA Implementation")
for i in np.logspace(strt, endd, count):
# for i in lengths:
    try:
        j += 1
        print(int(i))
        for ii in range(rep):
            torch.cuda.empty_cache()
            q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'), dtype=dtype,requires_grad=True)
            start_time = time.time()
            e = LA_torch(q,k,v,mask = mask,p = p)
            # e = torch.sum(fastmax_torch(q,k,v,mask = mask,p = p)).backward(retain_graph=True)
            print(e)
            cuda.synchronize()
            end_time = time.time()
            
            torch_LA_time[j] += (end_time - start_time)/rep
        torch_LA_memory[j] = torch.cuda.memory_allocated()
        # print(torch.cuda.memory_allocated())
    except:
        print("OOM for token length of ", int(i))

# print("\n softmax = \n")
# print("time")
temp = "["
for i in reg_attention_time: temp += str(i) + ", "
temp += "]"
print("Reg. Att. Time = ", temp)
# print("memory")
temp = "["
for i in soft_memory: temp += str(i) + ", "
temp += "]"
print("Reg. Att. Memory = ", temp)
print()
# print("LA with our implementation = \n")
# print("time")
temp = "["
for i in fast_time_custom: temp += str(i) + ", "
temp += "]"
print("Our LA Time = ", temp)
# print("memory")
temp = "["
for i in fast_memory_custom: temp += str(i) + ", "
temp += "]"
print("Our LA Memory = ", temp)
print()
# print("\n LA with Pytorch implementation = \n")
# print("time")
temp = "["
for i in fast_time: temp += str(i) + ", "
temp += "]"
print("Pythorch LA Time = ", temp)
# print("memory")
temp = "["
for i in fast_memory: temp += str(i) + ", "
temp += "]"
print("Pytorch LA Memory = ", temp)
