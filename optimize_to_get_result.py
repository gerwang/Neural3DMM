import torch
import torch.optim as optim
import torch.nn as nn

def optimize_to_get_result(model, loss_fun, z_dim, device, targets, n_iter=1, output_loss=True):
    model.eval()
    
#     return torch.empty(0), model(targets)
    
    model.module.only_encode = True
    z = model(targets)
    model.module.only_encode = False
    
    model.module.only_decode = True
    z_param = nn.Parameter(z)
    optimizer = optim.LBFGS(params=[z_param])
    for it in range(n_iter):
        def closure():
            optimizer.zero_grad()
            outputs = model(z_param)
            loss = loss_fun(outputs, targets)
            loss.backward()
            if output_loss:
                print('loss ', loss.item())
            return loss
        optimizer.step(closure)
    outputs = model(z_param)
    
    model.module.only_decode = False
    return z_param.data, outputs
