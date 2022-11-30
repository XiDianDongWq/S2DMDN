import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from net_new import NBNet
from torch import nn
from pavia_dataset import My_dataset
from scipy.io import savemat

batchsz = 6
lr = 1e-3
epochs = 300
Cin = 102
Cout = 4
N = 4
ratio = 4


device = torch.device('cuda:0')
torch.manual_seed(12)

root = '//media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/ztz/model_guide/data/pavia'

train_data = My_dataset(root,'train')
train_dataloader = DataLoader(train_data, batch_size=batchsz, shuffle=True)

test_data = My_dataset(root,'test')
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

for_data = My_dataset(root,'for')
for_dataloader = DataLoader(for_data, batch_size=1, shuffle=False)



def main():
    model = NBNet(Cin,Cout,N,ratio).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon1 = nn.L1Loss()
    criteon2 = nn.MSELoss()
    best_loss0, best_loss, running_loss = 1000000000,100000000,100000000
    global_step = 0
    # for epoch in range(epochs):
    #     for step,(Pan,lrHS,ref) in enumerate(train_dataloader):
    #         Pan = Pan.type(torch.float).to(device)
    #         lrHS = lrHS.type(torch.float).to(device)
    #         ref = ref.type(torch.float).to(device)
    #         model.train()
    #         output, output1, output2 = model(lrHS, Pan)
    #         running_loss0 = criteon1(output, ref)
    #         running_loss1 = 0.1 * criteon2(output1, Pan)
    #         running_loss2 = 0.1 * criteon2(output2, lrHS)
    #         running_loss = running_loss0 + running_loss1 + running_loss2
    #         optimizer.zero_grad()
    #         running_loss.backward()
    #         optimizer.step()
    #     global_step += 1
    #     if epoch % 1 == 0:
    #         if running_loss <= best_loss:
    #             best_loss = running_loss
    #             torch.save(model.state_dict(),'best_all.mdl')
    #         print(global_step,best_loss,running_loss0, running_loss1,running_loss2)

    torch.cuda.empty_cache()

    model.load_state_dict(torch.load('best_all.mdl'))

    model.eval()
    test_loss = 0

    for test_step, (test_Pan, test_lrHS, test_ref) in enumerate(test_dataloader):
        test_Pan, test_lrHS, test_ref = test_Pan.type(torch.float).to(device), test_lrHS.type(torch.float).to(device), test_ref.type(torch.float).to(device)

        with torch.no_grad():
            test_output,Pan,LRHS = model(test_lrHS, test_Pan)
            print(criteon1(test_output,test_ref))
            # savemat("Out.mat" , {'x': test_output.detach().cpu().numpy()})
            # savemat("Pan.mat" , {'x': Pan.detach().cpu().numpy()})
            # savemat("LRHS.mat", {'x': LRHS.detach().cpu().numpy()})
            # savemat("X_tilde.mat", {'x':X_tilde.detach().cpu().numpy()})
            # savemat("Y_title.mat", {'x': Y_title.detach().cpu().numpy()})

if __name__ == '__main__':
    main()
