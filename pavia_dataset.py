from torch.utils.data import Dataset
from scipy.io import loadmat
import os



class My_dataset(Dataset):
    def __init__(self,root,mode):
        super(My_dataset,self).__init__()
        self.root = root
        self.mode = mode
        self.gtHS = []
        self.LRHS = []
        self.PAN = []
        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root,"train", "gtHS"))
            self.gtHS.sort(key = lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "train", "LRHS"))
            self.LRHS.sort(key = lambda x: int(x.split(".")[0]))
            self.PAN = os.listdir(os.path.join(self.root,"train", "hrMS"))
            self.PAN.sort(key = lambda x: int(x.split(".")[0]))

        if self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root,'test', "gtHS"))
            self.gtHS.sort(key = lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root,'test', "LRHS"))
            self.LRHS.sort(key = lambda x: int(x.split(".")[0]))
            self.PAN = os.listdir(os.path.join(self.root,'test', "hrMS"))
            self.PAN.sort(key = lambda x: int(x.split(".")[0]))
        if self.mode == "for":
            self.gtHS = os.listdir(os.path.join(self.root, 'for', "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, 'for', "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            self.PAN = os.listdir(os.path.join(self.root, 'for', "hrMS"))
            self.PAN.sort(key=lambda x: int(x.split(".")[0]))
    def  __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):

        gt_hs,lr_hs,pan = self.gtHS[index], self.LRHS[index], self.PAN[index]
        if self.mode == "for":
            data_ref = loadmat(os.path.join(self.root, self.mode, "gtHS", gt_hs))['da'].reshape(102, 300, 300)
            data_lrHS = loadmat(os.path.join(self.root, self.mode, "LRHS", lr_hs))['da'].reshape(102, 75, 75)
            data_Pan = loadmat(os.path.join(self.root, self.mode, "hrMS", pan))['hrMS'].reshape(4, 300, 300)
        else:
            data_ref = loadmat(os.path.join(self.root,self.mode,"gtHS",gt_hs))['da'].reshape(102,160,160)
            data_lrHS = loadmat(os.path.join(self.root,self.mode,"LRHS",lr_hs))['da'].reshape(102,40,40)
            data_Pan = loadmat(os.path.join(self.root,self.mode,"hrMS",pan))['hrMS'].reshape(4,160,160)


        return  data_Pan,data_lrHS,data_ref
