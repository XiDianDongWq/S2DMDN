import os
import numpy as np
import random
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
from sklearn.decomposition import PCA
import time
from net1 import Net
import cv2
# setting paths
DataPath1 = '2012_Huston.mat' 
DataPath2 = 'LIDAR.mat'
TRPath = 'trainlabels7.mat'
TSPath = 'testlabels.mat'

#Fixed random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# setting parameters
patchsize1 = 11
patchsize2 = 11
pad_width = np.floor(patchsize1/2)
batchsize =4
EPOCH = 300
LR = 0.0001
NC = 5
cls_num=15

# load data
TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)
TrLabel = TrLabel['trainlabels']
TsLabel = TsLabel['testlabels']

Data = io.loadmat(DataPath1)
Data = Data['spectraldata']
Data = Data.astype(np.float32)

Data2 = io.loadmat(DataPath2)
Data2 = Data2['LiDAR']
Data2 = Data2.astype(np.float32)


# Data standardization
def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)
Data = sample_wise_standardization(Data)
Data2 = sample_wise_standardization(Data2)


# extract the principal components
[m, n, l] = Data.shape
PC = np.reshape(Data, (m*n, l))
pca = PCA(n_components=NC, copy=True, whiten=False)
PC = pca.fit_transform(PC)
PC = np.reshape(PC, (m, n, NC))
io.savemat('pca.mat',{'a':PC})

# boundary interpolation
temp = PC[:,:,0]
pad_width = np.floor(patchsize1/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape

# x : PCA_data
x = np.empty((m2,n2,NC),dtype='float32')
for i in range(NC):
    temp = PC[:,:,i]
    pad_width = np.floor(patchsize1/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x[:,:,i] = temp2
# x2: LiDAR_data
x2 = Data2
pad_width2 = np.floor(patchsize2/2)
pad_width2 = np.int(pad_width2)
temp2 = np.pad(x2, pad_width2, 'symmetric')
x2 = temp2

# construct the training and testing set of HSI p*p

[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum*4, patchsize1, patchsize1, NC), dtype='float32')
TrainLabel = np.empty(TrainNum*4)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]

    TrainPatch[i, :, :, :] = patch
    TrainPatch[i+len(ind1), :, :, :] = np.flip(patch, axis=0)
    noise = np.random.normal(0.0, 0.01, size=patch.shape)
    TrainPatch[i+2*len(ind1), :, :, :] = np.flip(patch + noise, axis=1)
    k = np.random.randint(4)
    TrainPatch[i+3*len(ind1), :, :, :] = np.rot90(patch, k=k)
    patchlabel = TrLabel[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel
    

[ind1, ind2] = np.where(TsLabel != 0)
# print(ind1)
TestNum = len(ind1)
TestPatch = np.empty((TestNum, patchsize1, patchsize1, NC), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    
    TestPatch[i, :, :, :] = patch
  
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel


print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)

# construct the training and testing set of HSI 1*1
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch3 = np.empty((TrainNum*4, 1, 1, 144), dtype='float32')
TrainLabel3= np.empty(TrainNum*4)
ind3 = ind1
ind4 = ind2
for i in range(len(ind1)):
    patch = Data[ind3[i], ind4[i], :]

    TrainPatch3[i, :, :, :] = patch
    TrainPatch3[i+len(ind1), :, :, :] = patch
    noise = np.random.normal(0.0, 0.01, size=patch.shape)
    TrainPatch3[i+2*len(ind1), :, :, :] = patch + noise
    TrainPatch3[i+3*len(ind1), :, :, :] = patch
    patchlabel3 = TrLabel[ind1[i], ind2[i]]
    TrainLabel3[i] = patchlabel3
    TrainLabel3[i+len(ind1)] = patchlabel3
    TrainLabel3[i+2*len(ind1)] = patchlabel3
    TrainLabel3[i+3*len(ind1)] = patchlabel3

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch3 = np.empty((TestNum, 1,1, 144), dtype='float32')
TestLabel3 = np.empty(TestNum)
ind3 = ind1
ind4 = ind2
for i in range(len(ind1)):
    patch = Data[ind3[i], ind4[i], :]
    # patch = np.reshape(patch, (1, 144))
    # patch = np.transpose(patch)
    # patch = np.reshape(patch, (144, 1, 1))
    TestPatch3[i, :, :, :] = patch


    patchlabel3 = TsLabel[ind1[i], ind2[i]]
    TestLabel3[i] = patchlabel3


# construct the training and testing set of LiDAR
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch2 = np.empty((TrainNum*4, patchsize2, patchsize2, 1), dtype='float32')
TrainLabel2 = np.empty(TrainNum*4)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    # print(patch.shape)
    # patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    # patch = np.transpose(patch)
    # patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TrainPatch2[i, :, :, 0] = patch
    TrainPatch2[i+len(ind1), :, :, 0] = np.flip(patch, axis=0)
    noise = np.random.normal(0.0, 0.01, size=patch.shape)
    TrainPatch2[i+2*len(ind1), :, :, 0] = np.flip(patch + noise, axis=1)
    k = np.random.randint(4)
    TrainPatch2[i+3*len(ind1), :, :, 0] = np.rot90(patch, k=k)    
    
    patchlabel2 = TrLabel[ind1[i], ind2[i]]
    TrainLabel2[i] = patchlabel2
    TrainLabel2[i+len(ind1)] = patchlabel2
    TrainLabel2[i+2*len(ind1)] = patchlabel2
    TrainLabel2[i+3*len(ind1)] = patchlabel2
    
    
[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch2 = np.empty((TestNum, patchsize2, patchsize2, 1), dtype='float32')
TestLabel2 = np.empty(TestNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    # patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    # patch = np.transpose(patch)
    # patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TestPatch2[i, :, :, 0] = patch    
    patchlabel2 = TsLabel[ind1[i], ind2[i]] 
    TestLabel2[i] = patchlabel2


print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)

# step3: change data to the input type of PyTorch
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainPatch1 = TrainPatch1.permute(0,3,1,2)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()
# dataset1 = dataf.TensorDataset(TrainPatch1, TrainLabel1)
# train_loader1 = dataf.DataLoader(dataset1, batch_size=batchsize, shuffle=True)

TestPatch1 = torch.from_numpy(TestPatch)
TestPatch1 =TestPatch1.permute(0,3,1,2)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()


Classes = len(np.unique(TrainLabel))

TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainPatch2 = TrainPatch2.permute(0,3,1,2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()

TrainPatch3 = torch.from_numpy(TrainPatch3)
TrainPatch3 = TrainPatch3.permute(0,3,1,2)
dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2,TrainPatch3)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)


TestPatch2 = torch.from_numpy(TestPatch2)
TestPatch2 = TestPatch2.permute(0,3,1,2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel2.long()

TestPatch3 = torch.from_numpy(TestPatch3)
TestPatch3 = TestPatch3.permute(0,3,1,2)
TestLabel3 = torch.from_numpy(TestLabel3)-1
TestLabel3 = TestLabel3.long()


# kappa
def kappa(cls_num,labels,predicted):
    conf_mat = np.zeros([cls_num, cls_num])
    for i in range(len(labels)):
    	true_i = np.array(labels[i])
    	# print(true_i)
    	pre_i = np.array(predicted[i])
    	# print(pre_i)
    	conf_mat[true_i, pre_i] += 1.0
    pe_rows = np.sum(conf_mat, axis=0)
    # print(pe_rows)
    pe_cols = np.sum(conf_mat, axis=1)
    sum_total = np.sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(conf_mat) / float(sum_total)
    kappa = (po - pe) / (1 - pe)
    return kappa
# loss
def CrossEntropyLoss_label_smooth(outputs, targets, num_classes=15, epsilon=0.1):
    # outputs = outputs.permute(0,2,3,1).reshape(-1,15)
    # targets = targets.reshape(-1)
    #print(outputs[:10,:,...])
    N = targets.size(0)
    # torch.Size([8, 10])
    # 初始化一个矩阵, 里面的值都是epsilon / (num_classes - 1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1)).cuda()

    targets = targets.data
    # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon).cuda()
    #print(smoothed_labels[:10,...])
    # 调用torch的log_softmax
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    # 用之前得到的smoothed_labels来调整log_prob中每个值
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


# cnn = Net()
cnn = Net(batchsize)
# print('The structure of the designed network', cnn)

# move model to GPU
cnn.cuda()

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
labels=TestLabel1.numpy()
BestAcc = 0
best_loss = 10
best_accuracy = 0
torch.cuda.synchronize()
start = time.time()
# train and test the designed model
# for epoch in range(EPOCH):
#     for step, (b_x1, b_x2, b_y,b_x3) in enumerate(train_loader):
#
#         # move train data to GPU
#         b_x1 = b_x1.cuda()
#         b_x2 = b_x2.cuda()
#         b_y = b_y.cuda()
#         b_x3 = b_x3.cuda()
#
#         out1 = cnn(b_x1, b_x2,b_x3)
#         # print(out1.shape,b_y.shape)
#         loss = loss_func(out1, b_y)
#         # loss = CrossEntropyLoss_label_smooth(out1, b_y)
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
#
#     # if epoch % 1 == 0:
#     #     torch.save(cnn.state_dict(),'best%d.mdl' % (epoch))
#     #     print(epoch,loss,best_loss)
#     if epoch % 1 == 0:
#         cnn.eval()
#         pred_y = np.empty((len(TestLabel)), dtype='float32')
#         number = len(TestLabel) // 50
#         for i in range(number):
#             temp = TestPatch1[i * 50:(i + 1) * 50, :, :, :]
#             temp = temp.cuda()
#             temp1 = TestPatch2[i * 50:(i + 1) * 50, :, :, :]
#             temp1 = temp1.cuda()
#             temp4 = TestPatch3[i * 50:(i + 1) * 50, :, :, :]
#             temp4 = temp4.cuda()
#             temp2 = cnn(temp,temp1,temp4)
#             temp3 = torch.max(temp2, 1)[1].squeeze()
#             pred_y[i * 50:(i + 1) * 50] = temp3.cpu()
#             del temp, temp2, temp3,temp4
#
#         if (i + 1) * 50 < len(TestLabel):
#             temp = TestPatch1[(i + 1) * 50:len(TestLabel), :, :, :]
#             temp = temp.cuda()
#             temp1 = TestPatch2[(i + 1) * 50:len(TestLabel), :, :, :]
#             temp1 = temp1.cuda()
#             temp4 = TestPatch3[(i + 1) * 50:len(TestLabel), :, :, :]
#             temp4 = temp4.cuda()
#             temp2 = cnn(temp,temp1,temp4)
#             temp3 = torch.max(temp2, 1)[1].squeeze()
#             pred_y[(i + 1) * 50:len(TestLabel)] = temp3.cpu()
#             del temp, temp2, temp3,temp4
#
#         pred_y = torch.from_numpy(pred_y).long()
#         accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
#         # pred_y = np.empty((len(TestLabel)), dtype='float32')
#         predicted=pred_y.numpy()
#
#         KA = kappa(cls_num,labels,predicted)
#         if accuracy >= best_accuracy:
#             best_accuracy = accuracy
#             torch.save(cnn.state_dict(),'best.mdl')



            
torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start

# # test each class accuracy
# # divide test set into many subsets

# load the saved parameters
cnn.load_state_dict(torch.load('pca5.mdl'))
cnn.eval()

torch.cuda.synchronize()
start = time.time()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//50

for i in range(number):
    temp = TestPatch1[i * 50:(i + 1) * 50, :, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[i * 50:(i + 1) * 50, :, :, :]
    temp1 = temp1.cuda()
    temp4 = TestPatch3[i * 50:(i + 1) * 50, :, :, :]
    temp4 = temp4.cuda()            
    temp2 = cnn(temp,temp1,temp4)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i * 50:(i + 1) * 50] = temp3.cpu()
    del temp, temp2, temp3,temp4

if (i + 1) * 50 < len(TestLabel):
    temp = TestPatch1[(i + 1) * 50:len(TestLabel), :, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[(i + 1) * 50:len(TestLabel), :, :, :]
    temp1 = temp1.cuda()
    temp4 = TestPatch3[(i + 1) * 50:len(TestLabel), :, :, :]
    temp4 = temp4.cuda()            
    temp2 = cnn(temp,temp1,temp4)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i + 1) * 50:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3,temp4



pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()

# print(OA)
print(EachAcc)
print(np.sum(EachAcc)/15)

torch.cuda.synchronize()
end = time.time()
print(end - start)
Test_time = end - start
Final_OA = OA

print('The OA is: ', Final_OA)
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)


labels=TestLabel1.numpy()
predicted=pred_y.numpy()
# 第一步：创建混淆矩阵
# 获取类别数，创建 N*N 的零矩阵
conf_mat = np.zeros([cls_num, cls_num])
# 第二步：获取真实标签和预测标签
# labels 为真实标签，通常为一个 batch 的标签
# predicted 为预测类别，与 labels 同长度
# 第三步：依据标签为混淆矩阵计数
for i in range(len(labels)):
	true_i = np.array(labels[i])
	pre_i = np.array(predicted[i])
	conf_mat[true_i, pre_i] += 1.0


pe_rows = np.sum(conf_mat, axis=0)
print(pe_rows)
pe_cols = np.sum(conf_mat, axis=1)
sum_total = np.sum(pe_cols)
pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
po = np.trace(conf_mat) / float(sum_total)
kappa = (po - pe) / (1 - pe)
print('The kappa is: ', kappa)
print('The OA is: ', Final_OA)


label2color = torch.tensor([
    [0,0,0],
    [128, 0, 0], 
    [255, 0, 0],  
    [255, 0, 255],  
    [189, 170, 93],  
    [100, 255, 0],  
    [0,190,190],  
    [0,255,255], 
    [0,0,255],
    [60,150,240], 
    [120,170,46], 
    [126,46,143],
    [237,176,34],
    [217,84,28],
    [0,115,190],
    [230,230,230]
    
])

image = torch.zeros(349,1905,3)
[ind1, ind2] = np.where(TrLabel != 0)
[ind3, ind4] = np.where(TsLabel != 0)
print(len(TrainLabel1))
print(len(ind1))

# for i in range(len(ind3)):
#     image[ind3[i],ind4[i]] = label2color[TestLabel1[i]+1]
#
# image = image.type(torch.uint8).numpy()
#
# cv2.imwrite('test_label.jpg',image)

# img = io.loadmat('endnet.mat')
# tb_trento = img['result']

# # for i in range(len(ind3)):
# #     image[ind3[i],ind4[i]] = label2color[tb_trento[ind3[i],ind4[i]]]


for i in range(len(ind1)):
    image[ind1[i],ind2[i]] = label2color[TrainLabel1[i]+1]

# for i in range(len(ind3)):
#     image[ind3[i],ind4[i]] = label2color[TestLabel1[i]+1]
#

for i in range(len(ind3)):
    image[ind3[i],ind4[i]] = label2color[pred_y[i]+1]

image = image.type(torch.uint8).numpy()
# io.savemat('image.mat',{'a':image})

cv2.imwrite('image.jpg',image)

# cv2.imwrite('ref.jpg',image)

