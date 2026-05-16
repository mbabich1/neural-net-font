import torch
from torch import nn, tensor

class StubModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.layers = nn.ModuleList(...)
        # self.fns = nn.ModuleList(...)

    def forward(self, x):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
