import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and processing
def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

# Content and style loss layers
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# VGG network model with inserted loss layers
def get_model_and_losses(cnn, content_img, style_img):
    cnn = copy.deepcopy(cnn)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential().to(device)
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Main style transfer loop
def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(cnn, content_img, style_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()

            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss = {style_score.item():.4f}, Content Loss = {content_score.item():.4f}")

            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# File paths (corrected!)
content_image_path = r"C:\Users\Sachin m\OneDrive\Desktop\AI\port.png"
style_image_path = r"C:\Users\Sachin m\Pictures\Camera Roll\Screenshots\Screenshot 2025-07-21 205023.png"
output_image_path = r"C:\Users\Sachin m\OneDrive\Desktop\AI\output.jpg"

# Run the style transfer
def stylize(content_path, style_path, output_path, size=512):
    content = load_image(content_path, size)
    style = load_image(style_path, size)
    input_img = content.clone()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    output = run_style_transfer(cnn, content, style, input_img)

    final_img = im_convert(output)
    final_pil = Image.fromarray((final_img * 255).astype('uint8'))
    final_pil.save(output_path)
    print(f"\n✅ Stylized image saved to: {output_path}")

# Call the function
stylize(content_image_path, style_image_path, output_image_path)
