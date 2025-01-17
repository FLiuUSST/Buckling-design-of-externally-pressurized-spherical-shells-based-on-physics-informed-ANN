import pandas as pd
import numpy as np
import torch

pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)        

data1 = input('Enter value for R: ')
data2 = input('Enter value for H: ')
data3 = input('Enter value for r: ')
data4 = input('Enter value for t: ')
data5 = input('Enter value for phi: ')
data6 = input('Enter value for lambda: ')
data7 = input('Enter value for E: ')
data8 = input('Enter value for sigma: ')
data9 = input('Enter value for mu: ')
data10 = input('Enter value for delta: ')
data11 = input('Enter value for boundary: ')
data12 = input('Enter value for EAB: ')
data13 = input('Enter value for rhoEBC: ')
data14 = input('Enter value for PEBC: ')

data_list = [float(data1), float(data2), float(data3), float(data4), float(data5), float(data6), float(data7), float(data8), float(data9), float(data10), float(data11), float(data12), float(data13), float(data14)]

if len(data_list) != 14:
    print("input should contain 14 elements")
else:
    column_titles = "R, H, r, t, phi, lambda, E, sigma, mu, delta, boundary, EAB, rhoEBC, PEBC"
    column_titles = column_titles.replace(",", " ").split()

    df = pd.DataFrame([data_list], columns=column_titles)
    print("\n")
    
    print('**********************************************************')
    print("input_data: ")
    print(df)
    print('**********************************************************')

data_norm = df.copy()

log10000_cols = ['PEBC']
log100_cols = [ 'EAB']
log10_cols = ['r', 't', 'mu', 'delta', 'rhoEBC']
log_cols = ['H', 'phi', 'lambda', 'boundary']
logn10_cols = ['R', 'sigma']
logn1000_cols = ['E']

for c in log10000_cols:
    data_norm[c] = np.log10(data_norm[c]*10000)

for c in log100_cols:
    data_norm[c] = np.log10(data_norm[c]*100)

for c in log10_cols:
    data_norm[c] = np.log10(data_norm[c]*10)

for c in log_cols:
    data_norm[c] = np.log10(data_norm[c])

for c in logn10_cols:
    data_norm[c] = np.log10(data_norm[c]/10)

for c in logn1000_cols:
    data_norm[c] = np.log10(data_norm[c]/1000)

X_input = torch.from_numpy(data_norm.to_numpy()).float()    

class Net(torch.nn.Module):
    def __init__(self, n_features, n_output):
          super(Net, self).__init__()
          self.layer1 = torch.nn.Linear(n_features, 22)
          self.layer2 = torch.nn.Linear(22, 20)
          self.layer3 = torch.nn.Linear(20, 18)
          self.layer4 = torch.nn.Linear(18, 16)
          self.layer5 = torch.nn.Linear(16, n_output)

    def forward(self,x):
          x = self.layer1(x)
          x = torch.relu(x)
          x = self.layer2(x)
          x = torch.relu(x)
          x = self.layer3(x)
          x = torch.relu(x)
          x = self.layer4(x)
          x = torch.relu(x)
          x = self.layer5(x)
          return x
net = Net(14,1)
net = torch.load('weights.pth', weights_only=True)

Y_output = net(X_input)
output = np.power(10, Y_output.view(-1).detach().cpu().numpy())/10000

print('**********************************************************')
print('The results is: ', round(float(output[0]),2))
print('**********************************************************')