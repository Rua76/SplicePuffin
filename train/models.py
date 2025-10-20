import torch
from torch_fftconv import FFTConv1d
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import h5py

# ------------------------------------------------------------
#  DA model (Donor + Acceptor) and dataset
# ------------------------------------------------------------
class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.fp = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        self.ctr = 0
        # Get dataset length from the number of X entries
        self.length = len([k for k in self.data.keys() if k.startswith('X')])

    def __getitem__(self, idx):
        self.ctr += 1
        if self.ctr % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.fp, 'r', libver='latest')
        X = torch.Tensor(self.data['X' + str(idx)][:].T)
        Y = torch.Tensor(self.data['Y' + str(idx)][:].T)
        M = torch.Tensor(self.data['M' + str(idx)][:].T)
        # For compatibility with your training loop, we return a dummy coord
        Z = self.data['Z' + str(idx)][:].T
        decode_Z =[z.decode('utf-8') for z in Z]
        
        #coord =[z[0].decode('utf-8') for z in coord]
        return (X, Y, M, decode_Z)

    def __len__(self):
        return self.length

class SimpleNetModified_DA(nn.Module):
    def __init__(self, input_channels=4):
        super(SimpleNetModified_DA, self).__init__()
        self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        #self.conv_donor = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        #self.conv_acceptor = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        self.activation = nn.Softplus()

        # Separate deconv layers for donor and acceptor 
        self.deconv_donor = FFTConv1d(40, 1, kernel_size=601, padding=300) 
        self.deconv_acceptor = FFTConv1d(40, 1, kernel_size=601, padding=300)  

    def forward(self, x):

        y = self.conv(x)  # Shape: (batch_size, 40, 5000)
        #y_donor = self.conv_donor(x)  # Shape: (batch_size, 40, 5000)
        #y_acceptor = self.conv_acceptor(x)  # Shape: (batch_size, 40, 5000)
        
        # activation
        yact = self.activation(y)   # Shape: (batch_size, 40, 5000)
        #yact_donor = self.activation(y_donor) # Shape: (batch_size, 40, 5000)
        #yact_acceptor = self.activation(y_acceptor) # Shape: (batch_size, 40, 5000)
        
        # Separate predictions for donor and acceptor
        y_pred_donor = torch.sigmoid(self.deconv_donor(yact))  # Shape: (batch_size, 1, 5000)
        y_pred_acceptor = torch.sigmoid(self.deconv_acceptor(yact))  # Shape: (batch_size, 1, 5000)
        
        # Concatenate the outputs along channel dimension
        y_pred = torch.cat([y_pred_donor, y_pred_acceptor], dim=1)  # Shape: (batch_size, 2, 5000)
        
        # Crop the output to match label shape: 5000 -> 4000 by removing 500 from each side
        return y_pred[:, :, 500:-500]  # Final shape: (batch_size, 2, 4000)
    
# ------------------------------------------------------------
#  NDA model (3 channel label + SSE) and dataset
# ------------------------------------------------------------

class H5Dataset_NDA(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.fp = file_path
        self.data = h5py.File(file_path, 'r', libver='latest')
        self.ctr = 0
        # Get dataset length from the number of X entries
        self.length = len([k for k in self.data.keys() if k.startswith('X')])

    def __getitem__(self, idx):
        self.ctr += 1
        if self.ctr % 100000 == 0: # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.fp, 'r', libver='latest')
        X = torch.Tensor(self.data['X' + str(idx)][:].T)
        Y = torch.Tensor(self.data['Y' + str(idx)][:].T)
        # For compatibility with your training loop, we return a dummy coord
        coord = 0          # (chrom, start, end, strand)
        
        #coord =[z[0].decode('utf-8') for z in coord]
        return (X, Y, coord)

    def __len__(self):
        return self.length

class SimpleNetModified_NDA(nn.Module):
    def __init__(self, input_channels=4, output_channels=4):
        super(SimpleNetModified_NDA, self).__init__()

        self.conv = nn.Conv1d(input_channels, 40, kernel_size=51, padding=25)
        self.activation = nn.Sigmoid()

        # Separate deconv layers for labels (3 channels) and SSE (1 channel)
        self.deconv_labels = FFTConv1d(40, output_channels-1, kernel_size=601, padding=300)  # 3 channels
        self.deconv_SSE = FFTConv1d(40, 1, kernel_size=601, padding=300)  # 1 channel

    def forward(self, x):
        y = self.conv(x)  # Shape: (batch_size, 40, 5000)
        yact = self.activation(y) * y  # Shape: (batch_size, 40, 5000)
        
        # Separate predictions for labels and SSE
        y_pred_label = F.softmax(self.deconv_labels(yact), dim=1)  # Shape: (batch_size, 3, 5000)
        y_pred_SSE = torch.sigmoid(self.deconv_SSE(yact))  # Shape: (batch_size, 1, 5000)
        
        # Concatenate the outputs along channel dimension
        y_pred = torch.cat([y_pred_label, y_pred_SSE], dim=1)  # Shape: (batch_size, 4, 5000)
        
        # Crop the output to match label shape: 5000 -> 4000 by removing 500 from each side
        return y_pred[:, :, 500:-500]  # Final shape: (batch_size, 4, 4000)