#pylint:disable=C0115
#pylint:disable=C0116
#pylint:disable=C0103

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torch import nn
from torchsummary import summary
import torch

# Setting up seed
torch.manual_seed(seed=42)

### Preparing Data ###

# Loading dataset
potato_dataset = datasets.ImageFolder(root="/storage/emulated/0/Download/archive/PotatoDisease/")

# Class names
class_names = potato_dataset.classes
class_indices = potato_dataset.class_to_idx

# Transformation for training and test data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Splitting dataset
train_split = int(0.8*len(potato_dataset))
test_split = len(potato_dataset) - train_split

train_data, test_data = random_split(dataset=potato_dataset, lengths=[train_split, test_split])

# Applying transformations
train_data.dataset.transform = train_transforms
test_data.dataset.transform = test_transforms

# Splitting into batches
train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# Plotting images
#image, label = next(iter(train_dataloader))
#img = image[30].permute(1, 2, 0)
#lbl = label[30].item()

#plt.imshow(img)
#plt.title(class_names[lbl])
#plt.axis(False)
#plt.show()

### Building the model ###

class PotatoDiseaseClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Building CNN layer - 1
        self.conv_stack_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        # Building CNN layer - 2
        self.conv_stack_2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        # Building CNN layer - 3
        self.conv_stack_3 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
        padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        # Building MLP layer
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=64*28*28, out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features=len(class_names))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack_1(x)
        x = self.conv_stack_2(x)
        x = self.conv_stack_3(x)
        x = self.classifier(x)
        return x
        
# Creating instance for our model
model = PotatoDiseaseClassificationModel()

# Summary of the model
#print(summary(model=model, input_size=(3, 224, 224)))

#print(class_names[model(torch.randn(size=(1, 3, 224, 224))).argmax(dim=1).item()])

# Accuracy function
def accuracy_fn(y_true, y_preds):
    """
    Calculates accuracy score
    """
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc


# Set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.6)


# Training loop and testing l
EPOCHS = 50

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}\n---------")
    
    # Training loop
    train_loss = 0
    train_acc = 0
    
    model.train()
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        train_logits = model(X_train)
        loss = loss_fn(train_logits, y_train)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y_train, y_preds=train_logits.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch  % 10 == 0:
            print(f"Looked at: {batch*len(X_train)}/{len(train_dataloader.dataset)} samples.")

    # Average training loss and accuracy
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    # Testing loop
    test_loss = 0
    test_acc = 0
    
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_logits = model(X_test)
            test_loss += loss_fn(test_logits, y_test).item()
            test_acc += accuracy_fn(y_true=y_test, y_preds=test_logits.argmax(dim=1))
        
    # Avearge test loss and accuracy
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    
    # Printing what's happening
    print(f"Train loss: {train_loss:.4f}  |  Train acc: {train_acc:.2f}%")
    print(f"Test loss: {test_loss:.4f}  |  Test acc: {test_acc:.2f}%")