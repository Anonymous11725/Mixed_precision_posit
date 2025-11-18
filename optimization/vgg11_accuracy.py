# import pandas as pd
# from numba import jit, prange
# from itertools import product
# import torch
# import torch.nn as nn
# import numpy as np
# from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from datetime import datetime
# import csv
# import os
# import argparse

# #########################################
# # Utility
# #########################################
# def log(msg):
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# # ----------------------------
# # Highly Optimized Posit Arithmetic with LUT
# # ----------------------------
# @jit(nopython=True, cache=True)
# def clz_fast(x, width=32):
#     """Count leading zeros efficiently"""
#     if x == 0:
#         return width
#     count = 0
#     mask = 1 << (width - 1)
#     while (x & mask) == 0:
#         count += 1
#         mask >>= 1
#     return count

# @jit(nopython=True, cache=True)
# def decode_posit_optimized(posit_val, N=16, es=2):
#     """Ultra-fast posit decoding with bit manipulation"""
#     if posit_val == 0:
#         return 0.0
    
#     # Extract sign
#     sign = 1.0 if (posit_val >> (N - 1)) == 0 else -1.0
    
#     # Handle negative numbers
#     if sign < 0:
#         posit_val = ((~posit_val + 1) & ((1 << N) - 1))
    
#     # Count regime bits more efficiently
#     shifted = posit_val << 1  # Remove sign bit
#     first_bit = (shifted >> (N - 1)) & 1
    
#     if first_bit == 1:
#         # Count leading 1s
#         leading_ones = clz_fast(~shifted << (32 - N), 32) if N < 32 else 0
#         k = leading_ones
#         regime_len = leading_ones + 1
#     else:
#         # Count leading 0s
#         leading_zeros = clz_fast(shifted << (32 - N), 32) if N < 32 else 0
#         k = -leading_zeros - 1
#         regime_len = leading_zeros + 1
    
#     # Extract exponent
#     exp_start = N - 1 - regime_len
#     exp_val = 0
#     if exp_start >= es:
#         exp_mask = ((1 << es) - 1) << (exp_start - es)
#         exp_val = (posit_val & exp_mask) >> (exp_start - es)
    
#     # Extract fraction
#     frac_start = exp_start - es
#     frac_val = 1.0
#     if frac_start > 0:
#         frac_mask = (1 << frac_start) - 1
#         frac_bits = posit_val & frac_mask
#         for i in range(frac_start):
#             if (frac_bits >> (frac_start - 1 - i)) & 1:
#                 frac_val += 2.0 ** -(i + 1)
    
#     # Compute final value
#     useed_power = k * (1 << es) + exp_val
#     return sign * (2.0 ** useed_power) * frac_val

# @jit(nopython=True, cache=True)
# def encode_posit_optimized(value, N=16, es=2):
#     """Ultra-fast posit encoding"""
#     if value == 0.0:
#         return 0
    
#     sign_bit = 0 if value > 0 else 1
#     value = abs(value)
    
#     # Find exponent
#     if value >= 1.0:
#         exp = 0
#         temp = value
#         while temp >= 2.0:
#             temp /= 2.0
#             exp += 1
#     else:
#         exp = 0
#         temp = value
#         while temp < 1.0:
#             temp *= 2.0
#             exp -= 1
    
#     # Split exponent into regime and exp parts
#     k = exp >> es
#     e = exp & ((1 << es) - 1)
    
#     # Normalize mantissa
#     if exp >= 0:
#         mantissa = value / (2.0 ** exp)
#     else:
#         mantissa = value * (2.0 ** (-exp))
    
#     # Build posit
#     result = 0
#     pos = N - 1
    
#     # Sign bit
#     if sign_bit:
#         result |= 1 << pos
#     pos -= 1
    
#     # Regime bits
#     if k >= 0:
#         regime_len = min(k + 1, pos + 1)
#         for i in range(regime_len):
#             result |= 1 << (pos - i)
#         pos -= regime_len
#         if pos >= 0:
#             pos -= 1  # Terminating bit
#     else:
#         regime_len = min(-k, pos + 1)
#         pos -= regime_len
#         if pos >= 0:
#             result |= 1 << pos  # Terminating bit
#             pos -= 1
    
#     # Exponent bits
#     for i in range(es):
#         if pos >= 0:
#             if (e >> (es - 1 - i)) & 1:
#                 result |= 1 << pos
#             pos -= 1
    
#     # Fraction bits
#     frac = mantissa - 1.0
#     for _ in range(pos + 1):
#         if pos >= 0:
#             frac *= 2.0
#             if frac >= 1.0:
#                 result |= 1 << pos
#                 frac -= 1.0
#             pos -= 1
    
#     # Apply two's complement for negative numbers
#     if sign_bit:
#         result = ((~result + 1) & ((1 << N) - 1))
    
#     return result

# @jit(nopython=True, cache=True)
# def posit_mac_optimized(a_val, b_val, acc_val, N=16, es=2):
#     """Optimized MAC with reduced decode/encode cycles"""
#     a_float = decode_posit_optimized(a_val, N, es)
#     b_float = decode_posit_optimized(b_val, N, es)
#     acc_float = decode_posit_optimized(acc_val, N, es)
    
#     result_float = a_float * b_float + acc_float
#     return encode_posit_optimized(result_float, N, es)

# # ----------------------------
# # Simplified and faster posit operations
# # ----------------------------
# @jit(nopython=True, cache=True)
# def simple_posit_encode(value, N=16, es=2):
#     """Simplified posit encoding for better accuracy"""
#     if abs(value) < 1e-38:  # Treat very small values as zero
#         return 0
    
#     if abs(value) > 1e38:   # Clamp very large values
#         value = 1e38 if value > 0 else -1e38
    
#     # Use a simpler encoding that's more numerically stable
#     sign = 0 if value >= 0 else 1
#     abs_val = abs(value)
    
#     # Find the scale
#     scale = 0
#     temp = abs_val
#     while temp >= 2.0:
#         temp /= 2.0
#         scale += 1
#     while temp < 1.0:
#         temp *= 2.0
#         scale -= 1
    
#     # Simple bit packing
#     result = 0
#     if sign:
#         result |= 1 << (N - 1)
    
#     # Pack the scale and mantissa in remaining bits
#     remaining_bits = N - 1
#     scale_bits = min(8, remaining_bits // 2)  # Use half bits for scale
#     mantissa_bits = remaining_bits - scale_bits
    
#     # Encode scale (biased)
#     scale_biased = scale + (1 << (scale_bits - 1))
#     scale_biased = max(0, min((1 << scale_bits) - 1, scale_biased))
#     result |= scale_biased << mantissa_bits
    
#     # Encode mantissa
#     mantissa = temp - 1.0
#     for i in range(mantissa_bits):
#         mantissa *= 2.0
#         if mantissa >= 1.0:
#             result |= 1 << (mantissa_bits - 1 - i)
#             mantissa -= 1.0
    
#     return result

# @jit(nopython=True, cache=True)
# def simple_posit_decode(posit_val, N=16, es=2):
#     """Simplified posit decoding for better accuracy"""
#     if posit_val == 0:
#         return 0.0
    
#     sign = -1.0 if (posit_val >> (N - 1)) & 1 else 1.0
    
#     remaining_bits = N - 1
#     scale_bits = min(8, remaining_bits // 2)
#     mantissa_bits = remaining_bits - scale_bits
    
#     # Extract scale
#     scale_mask = ((1 << scale_bits) - 1) << mantissa_bits
#     scale_biased = (posit_val & scale_mask) >> mantissa_bits
#     scale = scale_biased - (1 << (scale_bits - 1))
    
#     # Extract mantissa
#     mantissa_mask = (1 << mantissa_bits) - 1
#     mantissa_bits_val = posit_val & mantissa_mask
    
#     mantissa = 1.0
#     for i in range(mantissa_bits):
#         if (mantissa_bits_val >> (mantissa_bits - 1 - i)) & 1:
#             mantissa += 2.0 ** -(i + 1)
    
#     return sign * mantissa * (2.0 ** scale)

# @jit(nopython=True, cache=True)
# def simple_posit_mac(a_val, b_val, acc_val, N=16, es=2):
#     """Simplified MAC operation"""
#     a = simple_posit_decode(a_val, N, es)
#     b = simple_posit_decode(b_val, N, es)
#     acc = simple_posit_decode(acc_val, N, es)
    
#     result = a * b + acc
#     return simple_posit_encode(result, N, es)

# # ----------------------------
# # Ultra-fast convolution with proper implementation
# # ----------------------------
# @jit(nopython=True, parallel=True, cache=True)
# def ultra_fast_conv(input_data, kernel, pe_configs, pad_h, pad_w):
#     """Ultra-fast convolution implementation"""
#     print("inside ultra fast conv")
#     H, W = input_data.shape
#     KH, KW = kernel.shape
#     print(H, W)
#     # Validate/normalize pe_configs length
#     total_pes = KH * KW
#     if pe_configs.shape[0] != total_pes:
#         # fallback: replicate first config
#         baseN, basees = pe_configs[0, 0], pe_configs[0, 1]
#         tmp = np.zeros((total_pes, 2), dtype=np.int32)
#         for i in range(total_pes):
#             tmp[i, 0] = baseN
#             tmp[i, 1] = basees
#         pe_configs = tmp

#     # Pad input
#     padded_h = H + 2 * pad_h
#     padded_w = W + 2 * pad_w
#     padded = np.zeros((padded_h, padded_w), dtype=np.float32)
#     padded[pad_h:pad_h+H, pad_w:pad_w+W] = input_data

#     out_h = H + 2*pad_h - KH + 1
#     out_w = W + 2*pad_w - KW + 1
#     output = np.zeros((out_h, out_w), dtype=np.float32)

#     # Pre-encode weights once (weight-stationary)
#     # For each PE position, encode kernel[m, n] at that PE's (N, es)
#     w_enc = np.zeros(total_pes, dtype=np.int64)  # store as int
#     for m in range(KH):
#         for n in range(KW):
#             idx = m * KW + n
#             N = int(pe_configs[idx, 0])
#             es = int(pe_configs[idx, 1])
#             w_enc[idx] = simple_posit_encode(kernel[m, n], N, es)

#     for i in prange(out_h):
#         for j in range(out_w):
#             # Initialize accumulator in the first PE's precision
#             curN = int(pe_configs[0, 0])
#             cures = int(pe_configs[0, 1])
#             acc = simple_posit_encode(0.0, curN, cures)

#             # Convolution sum
#             for m in range(KH):
#                 for n in range(KW):
#                     idx = m * KW + n
#                     N = int(pe_configs[idx, 0])
#                     es = int(pe_configs[idx, 1])

#                     # If PE precision changes, convert accumulator
#                     if (N != curN) or (es != cures):
#                         acc_float = simple_posit_decode(acc, curN, cures)
#                         acc = simple_posit_encode(acc_float, N, es)
#                         curN = N
#                         cures = es

#                     a = padded[i + m, j + n]
#                     a_enc = simple_posit_encode(a, N, es)
#                     b_enc = w_enc[idx]  # already encoded at (N, es)

#                     # MAC at this PE's precision
#                     acc = simple_posit_mac(a_enc, b_enc, acc, N, es)

#             # Decode accumulator (in its last precision)
#             output[i, j] = simple_posit_decode(acc, curN, cures)

#     return output

# def run_conv_with_fallback(input_tensor, conv_layer, pe_config_list, use_posit=True):
#     """Run convolution with posit or fallback to standard"""
#     print("inside conv fallback")
#     if not use_posit:
#         return conv_layer(input_tensor)
    
#     try:
#         input_np = input_tensor.squeeze(0).numpy().astype(np.float32)
#         W = conv_layer.weight.detach().numpy().astype(np.float32)
#         B = conv_layer.bias.detach().numpy().astype(np.float32) if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
        
#         OC, IC, KH, KW = W.shape
#         pad_h, pad_w = conv_layer.padding if hasattr(conv_layer, 'padding') else (0, 0)
        
#         pe_configs = np.array(pe_config_list, dtype=np.int32)
#         print(pe_configs)
#         # pe_configs = np.array(pe_config_list, dtype=np.int32).reshape(-1, 2)

        
#         out_maps = []
#         for oc in range(OC):
#             acc_map = None
#             for ic in range(IC):
#                 kernel = W[oc, ic]
#                 out = ultra_fast_conv(input_np[ic], kernel, pe_configs, pad_h, pad_w)
#                 acc_map = out if acc_map is None else acc_map + out
            
#             acc_map += B[oc]
#             out_maps.append(acc_map)
        
#         result = torch.tensor(np.stack(out_maps), dtype=torch.float32).unsqueeze(0)
        
#         # Sanity check
#         if torch.isnan(result).any() or torch.isinf(result).any():
#             log("NaN/Inf detected in posit result, using standard conv")
#             return conv_layer(input_tensor)
        
#         return result
        
#     except Exception as e:
#         log(f"Posit conv failed: {e}, using standard conv")
#         return conv_layer(input_tensor)

# #########################################
# # VGG11 Model (matching vgg11_train_new.py)
# #########################################
# class VGG11(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.relu3 = nn.ReLU(inplace=True)
        
#         self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.pool4 = nn.MaxPool2d(2, 2)

#         self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.relu5 = nn.ReLU(inplace=True)
        
#         self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.pool6 = nn.MaxPool2d(2, 2)

#         self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.relu7 = nn.ReLU(inplace=True)
        
#         self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#         self.relu8 = nn.ReLU(inplace=True)
#         self.pool8 = nn.MaxPool2d(2, 2)

#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(512, 4096)
#         self.relu_fc1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.relu_fc2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(4096, num_classes)

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#         x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
#         x = self.relu5(self.bn5(self.conv5(x)))
#         x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
#         x = self.relu7(self.bn7(self.conv7(x)))
#         x = self.pool8(self.relu8(self.bn8(self.conv8(x))))
#         x = self.gap(x)
#         x = self.flatten(x)
#         x = self.relu_fc1(self.fc1(x))
#         x = self.relu_fc2(self.fc2(x))
#         return self.fc3(x)

# def forward_features_with_posit(x, model, posit_conv_idxs, pe_config_list):
#     """Optimized forward pass with posit convolutions"""
#     conv_layers = [
#         (model.conv1, model.bn1, model.relu1, model.pool1),
#         (model.conv2, model.bn2, model.relu2, model.pool2),
#         (model.conv3, model.bn3, model.relu3, None),
#         (model.conv4, model.bn4, model.relu4, model.pool4),
#         (model.conv5, model.bn5, model.relu5, None),
#         (model.conv6, model.bn6, model.relu6, model.pool6),
#         (model.conv7, model.bn7, model.relu7, None),
#         (model.conv8, model.bn8, model.relu8, model.pool8)
#     ]
    
#     for idx, (conv, bn, relu, pool) in enumerate(conv_layers):
#         # Apply convolution (posit or standard)
#         if idx in posit_conv_idxs:
#             x = run_conv_with_fallback(x, conv, pe_config_list, use_posit=True)
#         else:
#             x = conv(x)
        
#         # Apply batch norm, relu, and pooling
#         x = bn(x)
#         x = relu(x)
#         if pool is not None:
#             x = pool(x)
    
#     return x

# #########################################
# # Dataset utilities (from vgg11_train_new.py)
# #########################################
# IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
# CIFAR_MEAN_100, CIFAR_STD_100 = (0.5,), (0.5,)
# SVHN_MEAN, SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

# def get_test_loader(dataset, data_root, batch_size=32, num_workers=4, img_size=None, download=True):
#     """Get test data loader for different datasets"""
#     ds = dataset.lower()
    
#     if ds in ("cifar10", "cifar100", "svhn"):
#         size = img_size or 32
#         test_tf = transforms.Compose([
#             transforms.Resize((size, size)),
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR_MEAN_100 if ds != "svhn" else SVHN_MEAN,
#                                 CIFAR_STD_100 if ds != "svhn" else SVHN_STD),
#         ])
        
#         if ds == "cifar10":
#             test_dataset = CIFAR10(data_root, train=False, download=download, transform=test_tf)
#             num_classes = 10
#         elif ds == "cifar100":
#             test_dataset = CIFAR100(data_root, train=False, download=download, transform=test_tf)
#             num_classes = 100
#         else:  # svhn
#             test_dataset = SVHN(data_root, split='test', download=download, transform=test_tf)
#             num_classes = 10
    
#     elif ds in ("imagewoof", "folder"):
#         # Expect ImageFolder layout: data_root/val/class_name/*.jpg
#         size = img_size or 224
#         test_tf = transforms.Compose([
#             transforms.Resize(int(size * 1.15)),
#             transforms.CenterCrop(size),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#         ])
        
#         val_dir = os.path.join(data_root, "val")
#         if not os.path.isdir(val_dir):
#             raise FileNotFoundError(f"Expected '{val_dir}' for ImageFolder dataset.")
        
#         test_dataset = ImageFolder(val_dir, transform=test_tf)
#         num_classes = len(test_dataset.classes)
    
#     else:
#         raise ValueError("dataset must be one of: svhn | imagewoof | cifar10 | cifar100 | folder")
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return test_loader, num_classes

# #########################################
# # Evaluation function
# #########################################
# def evaluate_vgg11_with_posit(model, test_loader, posit_conv_idxs, pe_config_list, num_samples, device='cpu'):
#     """Evaluate VGG11 with posit arithmetic on specified conv layers"""
#     model.eval()
#     correct, total = 0, 0
#     print("inside evaluate vgg11")
#     # Pre-compile JIT functions
#     dummy_val = simple_posit_encode(1.0, 16, 2)
#     _ = simple_posit_decode(dummy_val, 16, 2)
    
#     with torch.no_grad():
#         for sample_idx, (img, label) in enumerate(test_loader):
#             if sample_idx >= num_samples:
#                 break
                
#             img, label = img.to(device), label.to(device)
            
#             # Forward pass with posit convs
#             print(pe_config_list)
#             x = forward_features_with_posit(img, model, posit_conv_idxs, pe_config_list)
            
#             # Classifier
#             x = model.gap(x)
#             x = model.flatten(x)
#             x = model.relu_fc1(model.fc1(x))
#             x = model.relu_fc2(model.fc2(x))
#             x = model.fc3(x)
            
#             pred = x.argmax(1)
#             correct += (pred == label).sum().item()
#             total += label.size(0)

    
#     return correct, total

# #########################################
# # Main with enhanced optimization and multi-dataset support
# #########################################
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("VGG11 Posit Evaluation - Multi-Dataset")
#     parser.add_argument("--dataset", default="cifar10", 
#                         choices=["cifar10", "cifar100", "svhn", "imagewoof", "folder"])
#     parser.add_argument("--data_root", default="./data")
#     parser.add_argument("--weights_path", default="./models/vgg11.pth")
#     parser.add_argument("--num_samples", type=int, default=1000)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--img_size", type=int, default=None)
#     parser.add_argument("--posit_conv_idx", type=int, default=0, 
#                         help="Which conv layer to use posit arithmetic (0-7)")
#     parser.add_argument("--N", type=int, default=10, help="Posit N parameter")
#     parser.add_argument("--es", type=int, default=2, help="Posit es parameter")
    
#     args = parser.parse_args()
    
#     # Configuration
#     POSIT_CONV_IDXS = [args.posit_conv_idx]
#     # PE_CONFIG = [(args.N, args.es)] * 9  # 3x3 kernel: 9 PEs with same config

#     # PE_CONFIG = [(7, 2), (9, 2), (9, 2), (9, 2), (8, 2), (8, 2), (8, 2), (9, 2), (9, 2)]
#     PE_CONFIG = [(7, 2), (7, 2), (7, 2), (7, 2), (7, 2), (8, 2), (6, 2), (9, 2), (9, 2)]
    
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = "cpu"
#     log(f"Using device: {device}")
    
#     # Get test loader and number of classes
#     try:
#         test_loader, num_classes = get_test_loader(
#             args.dataset, args.data_root, args.batch_size, 
#             img_size=args.img_size, download=True
#         )
#         log(f"Dataset: {args.dataset} | Classes: {num_classes}")
#     except Exception as e:
#         log(f"Error loading dataset: {e}")
#         exit(1)
    
#     # Load model
#     log("Loading VGG11 model...")
#     model = VGG11(num_classes=num_classes).to(device)
#     try:
#         state = torch.load(args.weights_path, map_location=device)
#         model.load_state_dict(state)
#         log("Model loaded successfully")
#     except FileNotFoundError:
#         log(f"Model file {args.weights_path} not found. Using random weights for testing.")
    
#     model.eval()

#     # Test standard model first for comparison
#     log("Testing standard model for comparison...")
#     start_time = datetime.now()
#     correct_std, total_std = 0, 0
#     with torch.no_grad():
#         for sample_idx, (img, label) in enumerate(tqdm(test_loader, desc="Standard", total=min(1000, args.num_samples))):
#             if sample_idx >= min(1000, args.num_samples):
#                 break
#             img, label = img.to(device), label.to(device)
#             x = model(img)
#             pred = x.argmax(1)
#             correct_std += (pred == label).sum().item()
#             total_std += label.size(0)
    
#     std_time = (datetime.now() - start_time).total_seconds()
#     std_acc = 100.0 * correct_std / total_std
#     log(f"Standard model: {std_acc:.2f}% accuracy in {std_time:.2f}s")

#     # Test posit model
#     log(f"Running posit version: {PE_CONFIG}, posit convs={POSIT_CONV_IDXS}")
    
#     start_time = datetime.now()
#     correct, total = evaluate_vgg11_with_posit(
#         model, test_loader, POSIT_CONV_IDXS, PE_CONFIG, args.num_samples, device
#     )
#     end_time = datetime.now()
    
#     acc = 100.0 * correct / total
#     elapsed = (end_time - start_time).total_seconds()
    
#     log(f"Posit Accuracy: {acc:.2f}% (samples: {total})")
#     log(f"Total time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")
#     log(f"Throughput: {total/elapsed:.1f} samples/second")
#     log(f"Accuracy difference: {acc - std_acc:.2f}%")
    
#     if elapsed < std_time:
#         speedup = std_time / elapsed
#         log(f"Speedup: {speedup:.2f}x faster than standard")
#     else:
#         slowdown = elapsed / std_time
#         log(f"Slowdown: {slowdown:.2f}x slower than standard")

import pandas as pd
from numba import jit, prange
from itertools import product
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import csv
import os
import argparse

#########################################
# Utility
#########################################
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ----------------------------
# Highly Optimized Posit Arithmetic with LUT
# ----------------------------
@jit(nopython=True, cache=True)
def clz_fast(x, width=32):
    """Count leading zeros efficiently"""
    if x == 0:
        return width
    count = 0
    mask = 1 << (width - 1)
    while (x & mask) == 0:
        count += 1
        mask >>= 1
    return count

@jit(nopython=True, cache=True)
def decode_posit_optimized(posit_val, N=16, es=2):
    """Ultra-fast posit decoding with bit manipulation"""
    if posit_val == 0:
        return 0.0
    
    # Extract sign
    sign = 1.0 if (posit_val >> (N - 1)) == 0 else -1.0
    
    # Handle negative numbers
    if sign < 0:
        posit_val = ((~posit_val + 1) & ((1 << N) - 1))
    
    # Count regime bits more efficiently
    shifted = posit_val << 1  # Remove sign bit
    first_bit = (shifted >> (N - 1)) & 1
    
    if first_bit == 1:
        # Count leading 1s
        leading_ones = clz_fast(~shifted << (32 - N), 32) if N < 32 else 0
        k = leading_ones
        regime_len = leading_ones + 1
    else:
        # Count leading 0s
        leading_zeros = clz_fast(shifted << (32 - N), 32) if N < 32 else 0
        k = -leading_zeros - 1
        regime_len = leading_zeros + 1
    
    # Extract exponent
    exp_start = N - 1 - regime_len
    exp_val = 0
    if exp_start >= es:
        exp_mask = ((1 << es) - 1) << (exp_start - es)
        exp_val = (posit_val & exp_mask) >> (exp_start - es)
    
    # Extract fraction
    frac_start = exp_start - es
    frac_val = 1.0
    if frac_start > 0:
        frac_mask = (1 << frac_start) - 1
        frac_bits = posit_val & frac_mask
        for i in range(frac_start):
            if (frac_bits >> (frac_start - 1 - i)) & 1:
                frac_val += 2.0 ** -(i + 1)
    
    # Compute final value
    useed_power = k * (1 << es) + exp_val
    return sign * (2.0 ** useed_power) * frac_val

@jit(nopython=True, cache=True)
def encode_posit_optimized(value, N=16, es=2):
    """Ultra-fast posit encoding"""
    if value == 0.0:
        return 0
    
    sign_bit = 0 if value > 0 else 1
    value = abs(value)
    
    # Find exponent
    if value >= 1.0:
        exp = 0
        temp = value
        while temp >= 2.0:
            temp /= 2.0
            exp += 1
    else:
        exp = 0
        temp = value
        while temp < 1.0:
            temp *= 2.0
            exp -= 1
    
    # Split exponent into regime and exp parts
    k = exp >> es
    e = exp & ((1 << es) - 1)
    
    # Normalize mantissa
    if exp >= 0:
        mantissa = value / (2.0 ** exp)
    else:
        mantissa = value * (2.0 ** (-exp))
    
    # Build posit
    result = 0
    pos = N - 1
    
    # Sign bit
    if sign_bit:
        result |= 1 << pos
    pos -= 1
    
    # Regime bits
    if k >= 0:
        regime_len = min(k + 1, pos + 1)
        for i in range(regime_len):
            result |= 1 << (pos - i)
        pos -= regime_len
        if pos >= 0:
            pos -= 1  # Terminating bit
    else:
        regime_len = min(-k, pos + 1)
        pos -= regime_len
        if pos >= 0:
            result |= 1 << pos  # Terminating bit
            pos -= 1
    
    # Exponent bits
    for i in range(es):
        if pos >= 0:
            if (e >> (es - 1 - i)) & 1:
                result |= 1 << pos
            pos -= 1
    
    # Fraction bits
    frac = mantissa - 1.0
    for _ in range(pos + 1):
        if pos >= 0:
            frac *= 2.0
            if frac >= 1.0:
                result |= 1 << pos
                frac -= 1.0
            pos -= 1
    
    # Apply two's complement for negative numbers
    if sign_bit:
        result = ((~result + 1) & ((1 << N) - 1))
    
    return result

@jit(nopython=True, cache=True)
def posit_mac_optimized(a_val, b_val, acc_val, N=16, es=2):
    """Optimized MAC with reduced decode/encode cycles"""
    a_float = decode_posit_optimized(a_val, N, es)
    b_float = decode_posit_optimized(b_val, N, es)
    acc_float = decode_posit_optimized(acc_val, N, es)
    
    result_float = a_float * b_float + acc_float
    return encode_posit_optimized(result_float, N, es)

# ----------------------------
# Simplified and faster posit operations
# ----------------------------
@jit(nopython=True, cache=True)
def simple_posit_encode(value, N=16, es=2):
    """Simplified posit encoding for better accuracy"""
    if abs(value) < 1e-38:  # Treat very small values as zero
        return 0
    
    if abs(value) > 1e38:   # Clamp very large values
        value = 1e38 if value > 0 else -1e38
    
    # Use a simpler encoding that's more numerically stable
    sign = 0 if value >= 0 else 1
    abs_val = abs(value)
    
    # Find the scale
    scale = 0
    temp = abs_val
    while temp >= 2.0:
        temp /= 2.0
        scale += 1
    while temp < 1.0:
        temp *= 2.0
        scale -= 1
    
    # Simple bit packing
    result = 0
    if sign:
        result |= 1 << (N - 1)
    
    # Pack the scale and mantissa in remaining bits
    remaining_bits = N - 1
    scale_bits = min(8, remaining_bits // 2)  # Use half bits for scale
    mantissa_bits = remaining_bits - scale_bits
    
    # Encode scale (biased)
    scale_biased = scale + (1 << (scale_bits - 1))
    scale_biased = max(0, min((1 << scale_bits) - 1, scale_biased))
    result |= scale_biased << mantissa_bits
    
    # Encode mantissa
    mantissa = temp - 1.0
    for i in range(mantissa_bits):
        mantissa *= 2.0
        if mantissa >= 1.0:
            result |= 1 << (mantissa_bits - 1 - i)
            mantissa -= 1.0
    
    return result

@jit(nopython=True, cache=True)
def simple_posit_decode(posit_val, N=16, es=2):
    """Simplified posit decoding for better accuracy"""
    if posit_val == 0:
        return 0.0
    
    sign = -1.0 if (posit_val >> (N - 1)) & 1 else 1.0
    
    remaining_bits = N - 1
    scale_bits = min(8, remaining_bits // 2)
    mantissa_bits = remaining_bits - scale_bits
    
    # Extract scale
    scale_mask = ((1 << scale_bits) - 1) << mantissa_bits
    scale_biased = (posit_val & scale_mask) >> mantissa_bits
    scale = scale_biased - (1 << (scale_bits - 1))
    
    # Extract mantissa
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
# Ultra-fast convolution with proper implementation
# ----------------------------
@jit(nopython=True, parallel=True, cache=True)
def ultra_fast_conv(input_data, kernel, pe_configs, pad_h, pad_w):
    """Ultra-fast convolution implementation"""
    
    H, W = input_data.shape
    KH, KW = kernel.shape
    
    # Validate/normalize pe_configs length
    total_pes = KH * KW
    if pe_configs.shape[0] != total_pes:
        # fallback: replicate first config
        baseN, basees = pe_configs[0, 0], pe_configs[0, 1]
        tmp = np.zeros((total_pes, 2), dtype=np.int32)
        for i in range(total_pes):
            tmp[i, 0] = baseN
            tmp[i, 1] = basees
        pe_configs = tmp

    # Pad input
    padded_h = H + 2 * pad_h
    padded_w = W + 2 * pad_w
    padded = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded[pad_h:pad_h+H, pad_w:pad_w+W] = input_data

    out_h = H + 2*pad_h - KH + 1
    out_w = W + 2*pad_w - KW + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Pre-encode weights once (weight-stationary)
    # For each PE position, encode kernel[m, n] at that PE's (N, es)
    w_enc = np.zeros(total_pes, dtype=np.int64)  # store as int
    for m in range(KH):
        for n in range(KW):
            idx = m * KW + n
            N = int(pe_configs[idx, 0])
            es = int(pe_configs[idx, 1])
            w_enc[idx] = simple_posit_encode(kernel[m, n], N, es)

    for i in prange(out_h):
        for j in range(out_w):
            # Initialize accumulator in the first PE's precision
            curN = int(pe_configs[0, 0])
            cures = int(pe_configs[0, 1])
            acc = simple_posit_encode(0.0, curN, cures)

            # Convolution sum
            for m in range(KH):
                for n in range(KW):
                    idx = m * KW + n
                    N = int(pe_configs[idx, 0])
                    es = int(pe_configs[idx, 1])

                    # If PE precision changes, convert accumulator
                    if (N != curN) or (es != cures):
                        acc_float = simple_posit_decode(acc, curN, cures)
                        acc = simple_posit_encode(acc_float, N, es)
                        curN = N
                        cures = es

                    a = padded[i + m, j + n]
                    a_enc = simple_posit_encode(a, N, es)
                    b_enc = w_enc[idx]  # already encoded at (N, es)

                    # MAC at this PE's precision
                    acc = simple_posit_mac(a_enc, b_enc, acc, N, es)

            # Decode accumulator (in its last precision)
            output[i, j] = simple_posit_decode(acc, curN, cures)

    return output

def run_conv_with_fallback(input_tensor, conv_layer, pe_config_list, use_posit=True):
    """Run convolution with posit or fallback to standard"""

    if not use_posit:
        return conv_layer(input_tensor)
    
    try:
        input_np = input_tensor.squeeze(0).numpy().astype(np.float32)
        W = conv_layer.weight.detach().numpy().astype(np.float32)
        B = conv_layer.bias.detach().numpy().astype(np.float32) if conv_layer.bias is not None else np.zeros(W.shape[0], dtype=np.float32)
        
        OC, IC, KH, KW = W.shape
        pad_h, pad_w = conv_layer.padding if hasattr(conv_layer, 'padding') else (0, 0)
        
        pe_configs = np.array(pe_config_list, dtype=np.int32)
        # print(pe_config_list)
        # print(pe_configs)
        # exit()
        # pe_configs = np.array(pe_config_list, dtype=np.int32).reshape(-1, 2)

        
        out_maps = []
        for oc in range(OC):
            acc_map = None
            for ic in range(IC):
                kernel = W[oc, ic]
                # print(f"pe_configs: {pe_configs.shape}")
                # print(f"pe_configs dtype: {pe_configs.dtype}")
                # print(f"input_data shape before passing: {input_np.shape}")
                # print(f"kernel shape before passing: {kernel.shape}")
                # exit()
                out = ultra_fast_conv(input_np[ic], kernel, pe_configs, pad_h, pad_w)
                acc_map = out if acc_map is None else acc_map + out
            
            acc_map += B[oc]
            out_maps.append(acc_map)
        
        result = torch.tensor(np.stack(out_maps), dtype=torch.float32).unsqueeze(0)
        
        # Sanity check
        if torch.isnan(result).any() or torch.isinf(result).any():
            log("NaN/Inf detected in posit result, using standard conv")
            return conv_layer(input_tensor)
        
        return result
        
    except Exception as e:
        log(f"Posit conv failed: {e}, using standard conv")
        return conv_layer(input_tensor)

#########################################
# VGG11 Model (matching vgg11_train_new.py)
#########################################
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool8 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.pool8(self.relu8(self.bn8(self.conv8(x))))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        return self.fc3(x)

def forward_features_with_posit(x, model, posit_conv_idxs, pe_config_list):
    """Optimized forward pass with posit convolutions"""
    conv_layers = [
        (model.conv1, model.bn1, model.relu1, model.pool1),
        (model.conv2, model.bn2, model.relu2, model.pool2),
        (model.conv3, model.bn3, model.relu3, None),
        (model.conv4, model.bn4, model.relu4, model.pool4),
        (model.conv5, model.bn5, model.relu5, None),
        (model.conv6, model.bn6, model.relu6, model.pool6),
        (model.conv7, model.bn7, model.relu7, None),
        (model.conv8, model.bn8, model.relu8, model.pool8)
    ]
    
    for idx, (conv, bn, relu, pool) in enumerate(conv_layers):
        # Apply convolution (posit or standard)
        if idx in posit_conv_idxs:
            x = run_conv_with_fallback(x, conv, pe_config_list, use_posit=True)
        else:
            x = conv(x)
        
        # Apply batch norm, relu, and pooling
        x = bn(x)
        x = relu(x)
        if pool is not None:
            x = pool(x)
    
    return x

#########################################
# Dataset utilities (from vgg11_train_new.py)
#########################################
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR_MEAN_100, CIFAR_STD_100 = (0.5,), (0.5,)
SVHN_MEAN, SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

def get_test_loader(dataset, data_root, batch_size=32, num_workers=4, img_size=None, download=True):
    """Get test data loader for different datasets"""
    ds = dataset.lower()
    
    if ds in ("cifar10", "cifar100", "svhn"):
        size = img_size or 32
        test_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN_100 if ds != "svhn" else SVHN_MEAN,
                                CIFAR_STD_100 if ds != "svhn" else SVHN_STD),
        ])
        
        if ds == "cifar10":
            test_dataset = CIFAR10(data_root, train=False, download=download, transform=test_tf)
            num_classes = 10
        elif ds == "cifar100":
            test_dataset = CIFAR100(data_root, train=False, download=download, transform=test_tf)
            num_classes = 100
        else:  # svhn
            test_dataset = SVHN(data_root, split='test', download=download, transform=test_tf)
            num_classes = 10
    
    elif ds in ("imagewoof", "folder"):
        # Expect ImageFolder layout: data_root/val/class_name/*.jpg
        size = img_size or 224
        test_tf = transforms.Compose([
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        
        val_dir = os.path.join(data_root, "val")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Expected '{val_dir}' for ImageFolder dataset.")
        
        test_dataset = ImageFolder(val_dir, transform=test_tf)
        num_classes = len(test_dataset.classes)
    
    else:
        raise ValueError("dataset must be one of: svhn | imagewoof | cifar10 | cifar100 | folder")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader, num_classes

#########################################
# Evaluation function
#########################################
# def evaluate_vgg11_with_posit(model, test_loader, posit_conv_idxs, pe_config_list, num_samples, device='cpu'):
#     """Evaluate VGG11 with posit arithmetic on specified conv layers"""
#     model.eval()
#     correct, total = 0, 0

#     # Pre-compile JIT functions
#     dummy_val = simple_posit_encode(1.0, 16, 2)
#     _ = simple_posit_decode(dummy_val, 16, 2)
    
#     with torch.no_grad():
#         for sample_idx, (img, label) in enumerate(test_loader):
#             if sample_idx >= num_samples:
#                 break
                
#             img, label = img.to(device), label.to(device)
            
#             # Forward pass with posit convs
#             print(pe_config_list)
#             x = forward_features_with_posit(img, model, posit_conv_idxs, pe_config_list)
            
#             # Classifier
#             x = model.gap(x)
#             x = model.flatten(x)
#             x = model.relu_fc1(model.fc1(x))
#             x = model.relu_fc2(model.fc2(x))
#             x = model.fc3(x)
            
#             pred = x.argmax(1)
#             correct += (pred == label).sum().item()
#             total += label.size(0)

    
#     return correct, total

def evaluate_vgg11_with_posit(model, test_loader, posit_conv_idxs, pe_config_list, num_samples, device='cpu'):
    """Evaluate VGG11 with posit arithmetic on specified conv layers"""
    model.eval()
    correct, total = 0, 0
    
    # Pre-compile JIT functions
    dummy_val = simple_posit_encode(1.0, 16, 2)
    _ = simple_posit_decode(dummy_val, 16, 2)
    
    with torch.no_grad():
        for sample_idx, (img, label) in enumerate(test_loader):
            if sample_idx >= num_samples:
                break

            # Extract a single image from the batch
            img = img[0:1]  # Get the first image from the batch
            # print(f"input_data shape before passing: {img.shape}")  # Should print (1, 3, 224, 224)
            
            # Check and print kernel shape for debugging
            kernel = model.conv1.weight.detach().cpu().numpy()  # Example: using the first kernel
            # print(f"kernel shape before passing: {kernel.shape}")  # Should print (64, 3, 3, 3) for a Conv2d with 64 filters

            # Now process the single image (flattened to (3, 224, 224))
            img = img.squeeze(0)  # Convert to (3, 224, 224)

            # Pass the image through the model
            output = forward_features_with_posit(img, model, posit_conv_idxs, pe_config_list)
            output = model.gap(output)
            output = model.flatten(output)
            output = model.relu_fc1(model.fc1(output))
            output = model.relu_fc2(model.fc2(output))
            output = model.fc3(output)

            pred = output.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    
    return correct, total


#########################################
# Main with enhanced optimization and multi-dataset support
#########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("VGG11 Posit Evaluation - Multi-Dataset")
    parser.add_argument("--dataset", default="cifar10", 
                        choices=["cifar10", "cifar100", "svhn", "imagewoof", "folder"])
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--weights_path", default="./models/vgg11.pth")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--posit_conv_idx", type=int, default=0, 
                        help="Which conv layer to use posit arithmetic (0-7)")
    parser.add_argument("--N", type=int, default=10, help="Posit N parameter")
    parser.add_argument("--es", type=int, default=2, help="Posit es parameter")
    
    args = parser.parse_args()
    
    # Configuration
    POSIT_CONV_IDXS = [args.posit_conv_idx]
    PE_CONFIG = [(args.N, args.es)] * 9  # 3x3 kernel: 9 PEs with same config

    PE_CONFIG = [(10, 2), (8, 2), (10, 2), (10, 2), (8, 2), (8, 2), (10, 2), (9, 2), (10, 2)]
    # PE_CONFIG = [(9, 2), (8, 2), (6, 2), (7, 2), (7, 2), (9, 2), (6, 2), (7, 2), (7, 2)]
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    log(f"Using device: {device}")
    
    # Get test loader and number of classes
    try:
        test_loader, num_classes = get_test_loader(
            args.dataset, args.data_root, args.batch_size, 
            img_size=args.img_size, download=True
        )
        log(f"Dataset: {args.dataset} | Classes: {num_classes}")
    except Exception as e:
        log(f"Error loading dataset: {e}")
        exit(1)
    
    # Load model
    log("Loading VGG11 model...")
    model = VGG11(num_classes=num_classes).to(device)
    try:
        state = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state)
        log("Model loaded successfully")
    except FileNotFoundError:
        log(f"Model file {args.weights_path} not found. Using random weights for testing.")
    
    model.eval()

    # Test standard model first for comparison
    log("Testing standard model for comparison...")
    start_time = datetime.now()
    correct_std, total_std = 0, 0
    with torch.no_grad():
        for sample_idx, (img, label) in enumerate(tqdm(test_loader, desc="Standard", total=min(1000, args.num_samples))):
            if sample_idx >= min(1000, args.num_samples):
                break
            img, label = img.to(device), label.to(device)
            x = model(img)
            pred = x.argmax(1)
            correct_std += (pred == label).sum().item()
            total_std += label.size(0)
    
    std_time = (datetime.now() - start_time).total_seconds()
    std_acc = 100.0 * correct_std / total_std
    log(f"Standard model: {std_acc:.2f}% accuracy in {std_time:.2f}s")

    # Test posit model
    log(f"Running posit version: {PE_CONFIG}, posit convs={POSIT_CONV_IDXS}")
    
    start_time = datetime.now()
    correct, total = evaluate_vgg11_with_posit(
        model, test_loader, POSIT_CONV_IDXS, PE_CONFIG, args.num_samples, device
    )
    end_time = datetime.now()
    
    acc = 100.0 * correct / total
    elapsed = (end_time - start_time).total_seconds()
    
    log(f"Posit Accuracy: {acc:.2f}% (samples: {total})")
    log(f"Total time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")
    log(f"Throughput: {total/elapsed:.1f} samples/second")
    log(f"Accuracy difference: {acc - std_acc:.2f}%")
    
    if elapsed < std_time:
        speedup = std_time / elapsed
        log(f"Speedup: {speedup:.2f}x faster than standard")
    else:
        slowdown = elapsed / std_time
        log(f"Slowdown: {slowdown:.2f}x slower than standard")
