import torch
weight=torch.FloatTensor([1,1,1])
loss = torch.nn.MSELoss()
input = torch.FloatTensor([[2,1,0],[1,0,0]])
target = torch.FloatTensor([[0,1,2],[0,2,1]])
print(input)
print(target)

def MSELOSS(weight,predict,fact):
    loss=weight[0]*sum((predict[:,0]-fact[:,0])**2)
    for i in range(1,len(weight)):
        loss+=weight[i]*sum((predict[:,i]-fact[:,i])**2)
    return loss/6.
    		



print(MSELOSS(weight,input,target))
output=loss(input,target)
print(output)
