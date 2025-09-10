import time
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader, random_split

torch.backends.quantized.engine = 'x86'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = torchvision.datasets.MNIST("MNIST", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data  = torchvision.datasets.MNIST("MNIST", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_size = int(0.8 * len(train_data))
val_size   = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_dataloader  = DataLoader(test_data,     batch_size=64, shuffle=False)

print("訓練集大小:", train_size)
print("驗證集大小:", val_size)
print("測試集大小:", len(test_data))

class NetworkMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        from torch.ao.quantization import QuantStub, DeQuantStub
        self.quant = QuantStub()

        self.conv1  = Conv2d(1, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1  = nn.ReLU6(inplace=True)
        self.maxp1  = MaxPool2d(2, 2)

        self.conv2  = Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2  = nn.ReLU6(inplace=True)
        self.maxp2  = MaxPool2d(2, 2)

        self.flatten = Flatten()
        self.fc1    = Linear(64*7*7, 128)
        self.relu3  = nn.ReLU6(inplace=True)
        self.fc2    = Linear(128, 10)

        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxp2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.dequant(x)
        return x

model_fp = NetworkMNIST()




loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer1 = torch.optim.SGD(model_fp.parameters(), lr=0.01, momentum=0.9)

def fit_model(model, loss_fn, optimizer, epochs, train_loader, val_loader):
    for epoch in range(1, epochs+1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for step, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_seen += imgs.size(0)

            if step % 100 == 0:
                print(f"  [Train] step {step:4d}  loss={loss.item():.4f}")

        train_loss = total_loss / total_seen
        train_acc  = total_correct / total_seen

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(imgs)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_seen += imgs.size(0)

        val_loss /= val_seen
        val_acc   = val_correct / val_seen

        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4%}")
        print(f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4%}")


model_fp.to(DEVICE)
fit_model(model_fp, loss_fn, optimizer1 , 10, train_dataloader, val_dataloader)

from torch.ao.quantization import fuse_modules, get_default_qat_qconfig, prepare_qat, convert

model_fp.eval()

model_fused = fuse_modules(
    model_fp,
    [["conv1", "bn1"], ["conv2", "bn2"]],
    inplace=False
)

model_fused.qconfig = get_default_qat_qconfig('x86')

model_fused.train()
model_qat = prepare_qat(model_fused)

model_qat.to(DEVICE)
optimizer2 = torch.optim.SGD(model_qat.parameters(), lr=0.01, momentum=0.9)
fit_model(model_qat, loss_fn, optimizer2, 10, train_dataloader, val_dataloader)

model_qat.to('cpu').eval()
model_int8 = torch.ao.quantization.convert(model_qat)
torch.save(model_int8.state_dict(), "model_int8_final.pth")

print("\n=== INT8 模型結構 ===")
print(model_int8)

def evaluate(model, dataloader, desc=""):
    model.eval()
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    t1 = time.time()
    print(f"[{desc}] Accuracy: {correct/total:.4%}, Inference time: {t1 - t0:.3f}s")

evaluate(model_qat, test_dataloader, "FP32 baseline")
evaluate(model_int8, test_dataloader, "INT8 quantized")



torch.set_printoptions(threshold=float('inf'))


print("\nconv1 weight shape:", model_int8.conv1.weight().shape)
print("conv1 int8 weights:\n", torch.int_repr(model_int8.conv1.weight()))

print("\nconv1 bias shape:", model_int8.conv1.bias().shape)
print("conv1 int8 bias values:\n",model_int8.conv1.bias())
print("--------------------------------------------------------------------------------------------------------------------")

print("\nconv2 weight shape:", model_int8.conv2.weight().shape)
print("conv2 int8 weights:\n", torch.int_repr(model_int8.conv2.weight()))

print("\nconv2 bias shape:", model_int8.conv2.bias().shape)
print("conv2 int8 bias values:\n",model_int8.conv2.bias())



