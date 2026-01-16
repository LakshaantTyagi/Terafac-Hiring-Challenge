#  Same from Level 2
train_transform = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size]
)

# validation should NOT have augmentation
val_dataset.dataset.transform = test_transform

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# SE RESNET DEFINATION

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(SEBasicBlock, self).__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def se_resnet18(num_classes=10):
    model = ResNet(
        block=SEBasicBlock,
        layers=[2, 2, 2, 2]
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
se_model = se_resnet18(num_classes=10)

pretrained_dict = base_model.state_dict()
model_dict = se_model.state_dict()

pretrained_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and not k.startswith("fc.")
}

model_dict.update(pretrained_dict)
se_model.load_state_dict(model_dict)

for param in se_model.parameters():
    param.requires_grad = False

for param in se_model.layer4.parameters():
    param.requires_grad = True

for param in se_model.fc.parameters():
    param.requires_grad = True

se_model = se_model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, se_model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

num_epochs = 6

train_accs, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(se_model, train_loader)
    val_loss, val_acc = evaluate(se_model, val_loader)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate(se_model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

#confusion matrix
all_preds = []
all_labels = []

se_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = se_model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.show()

target_layer = se_model.layer4[-1]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam

def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)

    return img

se_model.eval()
gradcam = GradCAM(se_model, target_layer)

# Get one test image
images, labels = next(iter(test_loader))
image = images[0].unsqueeze(0).to(device)
label = labels[0]

# Generate CAM
cam = gradcam.generate(image)

# Resize CAM to image size
cam_resized = cv2.resize(cam, (224, 224))

# Prepare image
original_img = denormalize(images[0])
heatmap = cv2.applyColorMap(
    np.uint8(255 * cam_resized),
    cv2.COLORMAP_JET
)
heatmap = heatmap / 255.0

overlay = 0.4 * heatmap + 0.6 * original_img

# GRAD CAM
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cam_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")

plt.show()
