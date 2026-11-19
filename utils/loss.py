import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                logits,     # (B, 10)
                targets):   # (B,)
        _exp_all        = torch.exp(logits)
        _sum            = torch.sum(_exp_all, dim=1, keepdim=True)

        _fracture       = _exp_all / _sum
        correct_probs = _fracture.gather(1, targets.unsqueeze(1))

        loss_i          = -1 * torch.log(correct_probs)

        loss            = torch.mean(loss_i, dim=0)

        return loss

if __name__=="__main__":
    my_loss = CrossEntropyLoss()
    torch_loss = nn.CrossEntropyLoss()

    logits = torch.randn(128, 10, requires_grad=True)
    targets = torch.randint(0, 10, (128,))

    print(my_loss(logits, targets))
    print(torch_loss(logits, targets))