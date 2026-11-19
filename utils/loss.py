import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        logits = logits.float()
        shifted = logits - logits.max(dim=1, keepdim=True).values
        log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
        loss_i = -log_probs.gather(1, targets.unsqueeze(1))
        return loss_i.mean()    

if __name__=="__main__":
    my_loss = CrossEntropyLoss()
    torch_loss = nn.CrossEntropyLoss()

    logits = torch.randn(128, 10, requires_grad=True)
    targets = torch.randint(0, 10, (128,))

    print(my_loss(logits, targets))
    print(torch_loss(logits, targets))