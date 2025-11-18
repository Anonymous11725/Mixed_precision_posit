import torch
import torch.nn as nn
import numpy as np
from numba import jit, prange
from torchvision.datasets import MNIST, FashionMNIST, EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import os

#########################################
# Utility
#########################################
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ----------------------------
# Simplified Posit Operations
# ----------------------------
@jit(nopython=True, cache=True)
def simple_posit_encode(value, N=16, es=2):
    """Simplified posit encoding"""
    if abs(value) < 1e-38:
        return 0
    
    if abs(value) > 1e38:
        value = 1e38 if value > 0 else -1e38
    
    sign = 0 if value >= 0 else 1
    abs_val = abs(value)
    
    scale = 0
    temp = abs_val
    while temp >= 2.0:
        temp /= 2.0
        scale += 1
    while temp < 1.0:
        temp *= 2.0
        scale -= 1
    
    result = 0
    if sign:
        result |= 1 << (N - 1)
    
    remaining_bits = N - 1
    scale_bits = min(8, remaining_bits // 2)
    mantissa_bits = remaining_bits - scale_bits
    
    scale_biased = scale + (1 << (scale_bits - 1))
    scale_biased = max(0, min((1 << scale_bits) - 1, scale_biased))
    result |= scale_biased << mantissa_bits
    
    mantissa = temp - 1.0
    for i in range(mantissa_bits):
        mantissa *= 2.0
        if mantissa >= 1.0:
            result |= 1 << (mantissa_bits - 1 - i)
            mantissa -= 1.0
    
    return result

@jit(nopython=True, cache=True)
def simple_posit_decode(posit_val, N=16, es=2):
    """Simplified posit decoding"""
    if posit_val == 0:
        return 0.0
    
    sign = -1.0 if (posit_val >> (N - 1)) & 1 else 1.0
    
    remaining_bits = N - 1
    scale_bits = min(8, remaining_bits // 2)
    mantissa_bits = remaining_bits - scale_bits
    
    scale_mask = ((1 << scale_bits) - 1) << mantissa_bits
    scale_biased = (posit_val & scale_mask) >> mantissa_bits
    scale = scale_biased - (1 << (scale_bits - 1))
    
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa_bits_val = posit_val & mantissa_mask
    
    mantissa = 1.0
    for i in range(mantissa_bits):
        if (mantissa_bits_val >> (mantissa_bits - 1 - i)) & 1:
            mantissa += 2.0 ** -(i + 1)
    
    return sign * mantissa * (2.0 ** scale)

@jit(nopython=True, cache=True)
def simple_posit_mac(a_val, b_val, acc_val, N=16, es=2):
    """Simplified MAC operation"""
    a = simple_posit_decode(a_val, N, es)
    b = simple_posit_decode(b_val, N, es)
    acc = simple_posit_decode(acc_val, N, es)
    
    result = a * b + acc
    return simple_posit_encode(result, N, es)

# ----------------------------
# Convolution with posit
# ----------------------------
@jit(nopython=True, parallel=True, cache=True)
def ultra_fast_conv(input_data, kernel, pe_configs, pad_h, pad_w):
    """Ultra-fast convolution with graded PE precision"""
    H, W = input_data.shape
    KH, KW = kernel.shape
    
    total_pes = KH * KW
    if pe_configs.shape[0] != total_pes:
        baseN, basees = pe_configs[0, 0], pe_configs[0, 1]
        tmp = np.zeros((total_pes, 2), dtype=np.int32)
        for i in range(total_pes):
            tmp[i, 0] = baseN
            tmp[i, 1] = basees
        pe_configs = tmp

    padded_h = H + 2 * pad_h
    padded_w = W + 2 * pad_w
    padded = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded[pad_h:pad_h+H, pad_w:pad_w+W] = input_data

    out_h = H + 2*pad_h - KH + 1
    out_w = W + 2*pad_w - KW + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Pre-encode weights (weight-stationary)
    w_enc = np.zeros(total_pes, dtype=np.int64)
    for m in range(KH):
        for n in range(KW):
            idx = m * KW + n
            N = int(pe_configs[idx, 0])
            es = int(pe_configs[idx, 1])
            w_enc[idx] = simple_posit_encode(kernel[m, n], N, es)

    for i in prange(out_h):
        for j in range(out_w):
            curN = int(pe_configs[0, 0])
            cures = int(pe_configs[0, 1])
            acc = simple_posit_encode(0.0, curN, cures)

            for m in range(KH):
                for n in range(KW):
                    idx = m * KW + n
                    N = int(pe_configs[idx, 0])
                    es = int(pe_configs[idx, 1])

                    if (N != curN) or (es != cures):
                        acc_float = simple_posit_decode(acc, curN, cures)
                        acc = simple_posit_encode(acc_float, N, es)
                        curN = N
                        cures = es

                    a = padded[i + m, j + n]
                    a_enc = simple_posit_encode(a, N, es)
                    b_enc = w_enc[idx]

                    acc = simple_posit_mac(a_enc, b_enc, acc, N, es)

            output[i, j] = simple_posit_decode(acc, curN, cures)

    return output

def run_conv_with_posit(input_tensor, conv_layer, pe_config_list):
    """Run convolution with posit arithmetic"""
    try:
        input_np = input_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        W = conv_layer.weight.detach().cpu().numpy().astype(np.float32)
        B = conv_layer.bias.detach().cpu().numpy().astype(np.float32) if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
        
        OC, IC, KH, KW = W.shape
        pad_h, pad_w = conv_layer.padding if hasattr(conv_layer, 'padding') else (0, 0)
        
        pe_configs = np.array(pe_config_list, dtype=np.int32)
        
        out_maps = []
        for oc in range(OC):
            acc_map = None
            for ic in range(IC):
                kernel = W[oc, ic]
                out = ultra_fast_conv(input_np[ic], kernel, pe_configs, pad_h, pad_w)
                acc_map = out if acc_map is None else acc_map + out
            
            acc_map += B[oc]
            out_maps.append(acc_map)
        
        result = torch.tensor(np.stack(out_maps), dtype=torch.float32).unsqueeze(0)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            log("NaN/Inf detected, using standard conv")
            return conv_layer(input_tensor)
        
        return result
        
    except Exception as e:
        log(f"Posit conv failed: {e}, using standard conv")
        return conv_layer(input_tensor)

#########################################
# LeNet-5 Model
#########################################
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def forward_with_posit(model, x, posit_conv_indices, pe_config_list):
    """Forward pass with selective posit convolutions"""
    # Conv1 + ReLU + Pool
    if 0 in posit_conv_indices:
        x = run_conv_with_posit(x, model.conv1, pe_config_list)
    else:
        x = model.conv1(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 2)
    
    # Conv2 + ReLU + Pool
    if 1 in posit_conv_indices:
        x = run_conv_with_posit(x, model.conv2, pe_config_list)
    else:
        x = model.conv2(x)
    x = torch.relu(x)
    x = torch.max_pool2d(x, 2)
    
    # Fully connected layers
    x = torch.flatten(x, 1)
    x = torch.relu(model.fc1(x))
    x = torch.relu(model.fc2(x))
    x = model.fc3(x)
    
    return x

#########################################
# Dataset utilities
#########################################
def get_test_loader(dataset, data_root, batch_size=32, download=True):
    """Get test data loader for different datasets"""
    ds = dataset.lower()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if ds == "mnist":
        test_dataset = MNIST(data_root, train=False, download=download, transform=transform)
        num_classes = 10
    elif ds == "fmnist" or ds == "fashionmnist":
        test_dataset = FashionMNIST(data_root, train=False, download=download, transform=transform)
        num_classes = 10
    elif ds == "emnist":
        # EMNIST 'byclass' split has 62 classes (digits + upper + lower case letters)
        split = "letters" if ds == "emnist" else ds.split("-", 1)[1]
        tgt_tf = (lambda y: y - 1) if split == 'letters' else None
        
        train_dataset = EMNIST(data_root, split=split, train=True, download=download, transform=transform, target_transform=tgt_tf)
        test_dataset = EMNIST(data_root, split=split, train=False, download=download, transform=transform, target_transform=tgt_tf)
        # num_classes = 62
        num_classes = len(train_dataset.classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: mnist, fmnist, emnist")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader, num_classes


#########################################
# Evaluation function
#########################################
def evaluate_lenet5_with_posit(model, test_loader, posit_conv_indices, pe_config_list, num_samples, device='cpu'):
    """Evaluate LeNet-5 with posit arithmetic on specified conv layers"""
    model.eval()
    model = model.to(device)
    correct, total = 0, 0
    
    # Pre-compile JIT
    dummy_val = simple_posit_encode(1.0, 16, 2)
    _ = simple_posit_decode(dummy_val, 16, 2)
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            if total >= num_samples:
                break
                
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Process each image individually
            for i in range(imgs.size(0)):
                if total >= num_samples:
                    break
                
                img = imgs[i:i+1]
                label = labels[i:i+1]
                
                # Forward pass with posit convs
                output = forward_with_posit(model, img, posit_conv_indices, pe_config_list)
                
                pred = output.argmax(1)
                correct += (pred == label).sum().item()
                total += 1
    
    return correct, total

#########################################
# Main evaluation script
#########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("LeNet-5 Posit Evaluation")
    parser.add_argument("--dataset", default="mnist", 
                        choices=["mnist", "fmnist", "fashionmnist", "emnist"])
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--weights_path", default="./models/lenet5_mnist.pth")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--posit_conv_idx", type=int, default=0, 
                        help="Which conv layer to use posit (0=conv1, 1=conv2)")
    parser.add_argument("--N", type=int, default=10, help="Posit N parameter")
    parser.add_argument("--es", type=int, default=2, help="Posit es parameter")
    
    args = parser.parse_args()
    
    # Configuration
    POSIT_CONV_IDXS = [args.posit_conv_idx]
    # PE_CONFIG = [(args.N, args.es)] * 25  # 5x5 kernel: 25 PEs

    PE_CONFIG = [(8, 2), (8, 2), (7, 2), (8, 2), (7, 2), (8, 2), (8, 2), (8, 2), (8, 2), (7, 2), (7, 2), (7, 2), (7, 2), (7, 2), (7, 2), (7, 2), (8, 2), (7, 2), (8, 2), (7, 2), (7, 2), (7, 2), (8, 2), (8, 2), (7, 2)]
    
    device = "cpu"
    log(f"Using device: {device}")
    
    # Get test loader
    try:
        test_loader, num_classes = get_test_loader(
            args.dataset, args.data_root, args.batch_size, download=True
        )
        log(f"Dataset: {args.dataset} | Classes: {num_classes}")
    except Exception as e:
        log(f"Error loading dataset: {e}")
        exit(1)
    
    # Load model
    log("Loading LeNet-5 model...")
    model = LeNet5(num_classes=num_classes).to(device)
    try:
        state = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state)
        log("Model loaded successfully")
    except FileNotFoundError:
        log(f"Model file {args.weights_path} not found. Using random weights.")
    except Exception as e:
        log(f"Error loading model: {e}")
    
    model.eval()

    # Test standard model
    log("Testing standard model...")
    start_time = datetime.now()
    correct_std, total_std = 0, 0
    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Standard", total=args.num_samples//args.batch_size):
            if total_std >= args.num_samples:
                break
            img, label = img.to(device), label.to(device)
            output = model(img)
            pred = output.argmax(1)
            correct_std += (pred == label).sum().item()
            total_std += label.size(0)
            if total_std > args.num_samples:
                correct_std -= (total_std - args.num_samples)
                total_std = args.num_samples
    
    std_time = (datetime.now() - start_time).total_seconds()
    std_acc = 100.0 * correct_std / total_std
    log(f"Standard model: {std_acc:.2f}% accuracy in {std_time:.2f}s")

    # Test posit model
    log(f"Running posit version: {PE_CONFIG[:5]}... (5x5 kernel), posit convs={POSIT_CONV_IDXS}")
    
    start_time = datetime.now()
    correct, total = evaluate_lenet5_with_posit(
        model, test_loader, POSIT_CONV_IDXS, PE_CONFIG, args.num_samples, device
    )
    end_time = datetime.now()
    
    acc = 100.0 * correct / total
    elapsed = (end_time - start_time).total_seconds()
    
    log(f"Posit Accuracy: {acc:.2f}% (samples: {total})")
    log(f"Total time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")
    log(f"Throughput: {total/elapsed:.1f} samples/second")
    log(f"Accuracy difference: {acc - std_acc:.2f}%")