
#!/bin/bash

echo "Running CollapseGrammarOptimizer vGH1 enhanced test..."

# 設置虛擬測試環境
python3 - <<END
import torch
import torch.nn as nn
import argparse
from CollapseGrammarOptimizer_vGH1_0 import CollapseGrammarOptimizer_vGH1

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
args = parser.parse_args()

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = CollapseGrammarOptimizer_vGH1(model.parameters(), lr=args.lr)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

with open("loss_trace.txt", "w") as f:
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        f.write(f"{epoch+1},{loss.item():.6f}\n")
END

echo "Enhanced test complete. Loss trace saved to loss_trace.txt"
