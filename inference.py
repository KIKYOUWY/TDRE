import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from TDRE import TDRE


def load_image(image_path, img_size=256):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return img, transform(img).unsqueeze(0)


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TDRE(n_expert=3, top_k=3, inter_ch=3).to(device)

    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

    model.eval()
    orig_img, input_tensor = load_image(args.image, img_size=args.img_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        clean_tensor, logits_moe, enhance_tensor = model(input_tensor)

    clean_img = tensor_to_image(clean_tensor)
    enhance_img = tensor_to_image(enhance_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(clean_img)
    axes[1].set_title('Restored Image')
    axes[1].axis('off')

    axes[2].imshow(enhance_img)
    axes[2].set_title('Enhanced Image')
    axes[2].axis('off')

    plt.tight_layout()
    if args.save_path:
        plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weight/weight.pth')
    parser.add_argument('--image', type=str, default='example/test_foggy.jpg')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    main(args)