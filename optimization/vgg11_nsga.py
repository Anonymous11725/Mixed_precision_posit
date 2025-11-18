# # import numpy as np
# # import torch
# # import random
# # import csv
# # import os
# # import argparse
# # from datetime import datetime

# # from torch.utils.data import DataLoader
# # from torchvision import transforms
# # from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

# # # Import the modified VGG11 model and evaluation functions
# # from vgg11_accuracy import VGG11, evaluate_vgg11_with_posit, get_test_loader, log

# # from pymoo.algorithms.moo.nsga2 import NSGA2
# # from pymoo.core.problem import ElementwiseProblem
# # from pymoo.termination import get_termination
# # from pymoo.optimize import minimize
# # from pymoo.operators.sampling.rnd import IntegerRandomSampling
# # from pymoo.operators.crossover.pntx import PointCrossover
# # from pymoo.core.mutation import Mutation

# # # =========================================================
# # # Global Config
# # # =========================================================
# # POPULATION = 50
# # GENERATIONS = 20
# # SEED = 42
# # EVAL_SAMPLES = 1000          # how many images to evaluate per candidate
# # BATCH_SIZE = 32
# # MUTATION_RATE = 0.30

# # # Allowed posit pairs (index -> (N, es))
# # VALID_PAIRS = [(6, 2), (8, 2), (7, 2), (9, 2)]

# # # Dataset and model configuration
# # DEFAULT_DATASET = "cifar10"
# # DEFAULT_DATA_ROOT = "./data"
# # DEFAULT_WEIGHTS_PATH = "./models/vgg11_cifar10.pth"

# # # Which convs run with posit arithmetic (by index in VGG11 forward)
# # # 0=conv1, 1=conv2, 2=conv3, 3=conv4, 4=conv5, 5=conv6, 6=conv7, 7=conv8
# # POSIT_CONV_INDICES = [0]  # Start with first conv only

# # # =========================================================
# # # Hardware Cost Model (simple)
# # # =========================================================
# # def hardware_cost_graded(config):
# #     """
# #     Cost proxy: sum of N^2 per-PE, plus 10% coordination overhead.
# #     config: list of 9 tuples [(N, es), ...]
# #     """
# #     total = sum(N * N for (N, es) in config)
# #     return total * 1.10

# # def generation_callback(algorithm):
# #     """Callback function to track generation progress"""
# #     print(f"Generation {algorithm.n_gen} completed")

# # # =========================================================
# # # Custom Mutation (encourages local tweaks + occasional global jumps)
# # # =========================================================
# # class GradedPrecisionMutation(Mutation):
# #     def __init__(self, prob=0.3):
# #         super().__init__()
# #         self.prob = prob

# #     def _do(self, problem, X, **kwargs):
# #         X = X.astype(int)
# #         n_choices = len(VALID_PAIRS)
# #         for i in range(len(X)):
# #             # per-PE edits
# #             for j in range(X.shape[1]):
# #                 if np.random.rand() < self.prob:
# #                     if np.random.rand() < 0.7:  # local tweak
# #                         delta = np.random.choice([-1, 0, 1])
# #                         X[i, j] = int(np.clip(X[i, j] + delta, 0, n_choices - 1))
# #                     else:  # global jump
# #                         X[i, j] = np.random.randint(0, n_choices)
# #         return X

# # # =========================================================
# # # NSGA-II Problem (9 vars = 3x3 PEs)
# # # =========================================================
# # class VGG11GradedPEProblem(ElementwiseProblem):
# #     def __init__(self, model, dataloader, device, log_file):
# #         super().__init__(
# #             n_var=9,                 # 9 PEs for a 3x3 kernel
# #             n_obj=2,                 # minimize [-accuracy, cost]
# #             xl=0, xu=len(VALID_PAIRS) - 1,
# #             type_var=int
# #         )
# #         self.model = model
# #         self.loader = dataloader
# #         self.device = device
# #         self.log_file = log_file
# #         self.evals = 0

# #     def _evaluate(self, x, out, *args, **kwargs):
# #         # map indices -> actual (N, es)
# #         pe_config = [VALID_PAIRS[int(i)] for i in x]

# #         print(pe_config)
# #         # run VGG11 evaluation with posit arithmetic
# #         correct, total = evaluate_vgg11_with_posit(
# #             self.model, self.loader,
# #             POSIT_CONV_INDICES, pe_config,
# #             num_samples=EVAL_SAMPLES, device=self.device
# #         )
# #         acc = 100.0 * correct / total if total > 0 else 0.0
# #         cost = hardware_cost_graded(pe_config)

# #         # NSGA-II minimizes objectives → negate accuracy
# #         out["F"] = [-acc, cost]

# #         # log each evaluation
# #         with open(self.log_file, "a", newline="") as f:
# #             writer = csv.writer(f)
# #             # writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", pe_config])
# #             # writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", pe_config.tolist()])
# #             writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", pe_config.tolist()])


# #         self.evals += 1
# #         log(f"Evaluation {self.evals}: Accuracy={acc:.2f}%, Cost={cost:.2f}")

        

# # # =========================================================
# # # Dataset configuration mapping
# # # =========================================================
# # def get_dataset_config(dataset_name):
# #     """Get dataset-specific configuration"""
# #     ds = dataset_name.lower()
    
# #     if ds == "cifar10":
# #         return {
# #             'mean': (0.4914, 0.4822, 0.4465),
# #             'std': (0.2023, 0.1994, 0.2010),
# #             'num_classes': 10,
# #             'img_size': 32
# #         }
# #     elif ds == "cifar100":
# #         return {
# #             'mean': (0.4914, 0.4822, 0.4465),
# #             'std': (0.2023, 0.1994, 0.2010),
# #             'num_classes': 100,
# #             'img_size': 32
# #         }
# #     elif ds == "svhn":
# #         return {
# #             'mean': (0.4377, 0.4438, 0.4728),
# #             'std': (0.198, 0.201, 0.197),
# #             'num_classes': 10,
# #             'img_size': 32
# #         }
# #     elif ds in ("imagewoof", "folder"):
# #         return {
# #             'mean': (0.485, 0.456, 0.406),
# #             'std': (0.229, 0.224, 0.225),
# #             'num_classes': None,  # Will be determined from dataset
# #             'img_size': 224
# #         }
# #     else:
# #         raise ValueError(f"Unknown dataset: {dataset_name}")

# # # =========================================================
# # # Main optimization function
# # # =========================================================
# # def run_vgg11_nsga2_optimization(
# #     dataset="cifar10",
# #     data_root="./data",
# #     weights_path="./models/vgg11.pth",
# #     img_size=None,
# #     population=50,
# #     generations=20,
# #     eval_samples=1000,
# #     batch_size=32,
# #     mutation_rate=0.3,
# #     posit_conv_indices=None,
# #     output_dir="./nsga2_results",
# #     seed=42
# # ):
# #     """Run NSGA-II optimization for VGG11 posit precision"""
    
# #     # Setup
# #     if posit_conv_indices is None:
# #         posit_conv_indices = [0]
    
# #     global POSIT_CONV_INDICES, EVAL_SAMPLES, BATCH_SIZE
# #     POSIT_CONV_INDICES = posit_conv_indices
# #     EVAL_SAMPLES = eval_samples
# #     BATCH_SIZE = batch_size
    
# #     # Set seeds
# #     torch.manual_seed(seed)
# #     random.seed(seed)
# #     np.random.seed(seed)

# #     device = "cpu"
# #     log(f"Using device: {device}")

# #     # Get dataset configuration
# #     dataset_config = get_dataset_config(dataset)
# #     if img_size is None:
# #         img_size = dataset_config['img_size']

# #     # Load dataset
# #     try:
# #         test_loader, num_classes = get_test_loader(
# #             dataset, data_root, batch_size, img_size=img_size, download=True
# #         )
# #         log(f"Dataset: {dataset} | Classes: {num_classes}")
# #     except Exception as e:
# #         log(f"Error loading dataset: {e}")
# #         return None

# #     # Load model
# #     model = VGG11(num_classes=num_classes).to(device)
# #     try:
# #         state = torch.load(weights_path, map_location=device)
# #         model.load_state_dict(state)
# #         log("Model loaded successfully")
# #     except FileNotFoundError:
# #         log(f"Model file {weights_path} not found. Using random weights.")
# #     except Exception as e:
# #         log(f"Error loading model: {e}")
# #         return None

# #     model.eval()

# #     # Create output directory
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # Setup logging
# #     log_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_log.csv")
# #     with open(log_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["time", "accuracy", "hardware_cost", "pe_config(9)"])

# #     # Setup NSGA-II problem
# #     # problem = VGG11GradedPEProblem(model, test_loader, device, log_file)
# #     print("problem start")
# #     problem = VGG11GradedPEProblem(model, test_loader, device, log_file)
# #     print("problem end")
# #     algorithm = NSGA2(
# #         pop_size=population,
# #         sampling=IntegerRandomSampling(),
# #         crossover=PointCrossover(n_points=2),
# #         mutation=GradedPrecisionMutation(prob=mutation_rate),
# #         eliminate_duplicates=True
# #     )

# #     log(f"Starting NSGA-II optimization:")
# #     log(f"  Population: {population}")
# #     log(f"  Generations: {generations}")
# #     log(f"  Evaluation samples: {eval_samples}")
# #     log(f"  Posit conv indices: {posit_conv_indices}")
# #     log(f"  Valid posit pairs: {VALID_PAIRS}")

# #     # Run optimization
# #     res = minimize(
# #         problem, algorithm,
# #         termination=get_termination("n_gen", generations),
# #         seed=seed, verbose=True,
# #         callback=generation_callback
# #     )

# #     # Save and display results
# #     results_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_pareto.csv")
# #     with open(results_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["accuracy", "hardware_cost", "pe_config"])
        
# #         log("\n" + "="*60)
# #         log("PARETO FRONT RESULTS:")
# #         log("="*60)
        
# #         for i, (x, f) in enumerate(zip(res.X, res.F)):
# #             cfg = [VALID_PAIRS[int(j)] for j in x]
# #             acc = -f[0]  # Convert back from minimization
# #             cost = f[1]
            
# #             writer.writerow([f"{acc:.2f}", f"{cost:.2f}", str(cfg)])
# #             log(f"Solution {i+1:2d}: Acc={acc:6.2f}%  Cost={cost:8.2f}  Config={cfg}")

# #     log(f"\nOptimization completed!")
# #     log(f"Results saved to: {results_file}")
# #     log(f"Full log saved to: {log_file}")
    
# #     return res

# # # =========================================================
# # # Command Line Interface
# # # =========================================================
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser("VGG11 NSGA-II Posit Precision Optimization")
    
# #     # Dataset arguments
# #     parser.add_argument("--dataset", default="cifar10",
# #                         choices=["cifar10", "cifar100", "svhn", "imagewoof", "folder"],
# #                         help="Dataset to use for optimization")
# #     parser.add_argument("--data_root", default="./data",
# #                         help="Root directory for dataset")
# #     parser.add_argument("--weights_path", default="./models/vgg11.pth",
# #                         help="Path to trained VGG11 model weights")
# #     parser.add_argument("--img_size", type=int, default=None,
# #                         help="Image size (default: 32 for CIFAR/SVHN, 224 for ImageFolder)")
    
# #     # Optimization arguments
# #     parser.add_argument("--population", type=int, default=50,
# #                         help="NSGA-II population size")
# #     parser.add_argument("--generations", type=int, default=20,
# #                         help="Number of generations")
# #     parser.add_argument("--eval_samples", type=int, default=1000,
# #                         help="Number of samples to evaluate per candidate")
# #     parser.add_argument("--batch_size", type=int, default=32,
# #                         help="Batch size for evaluation")
# #     parser.add_argument("--mutation_rate", type=float, default=0.3,
# #                         help="Mutation probability")
    
# #     # Posit configuration
# #     parser.add_argument("--posit_conv_indices", nargs="+", type=int, default=[0],
# #                         help="Indices of conv layers to use posit arithmetic (0-7)")
    
# #     # Output and misc
# #     parser.add_argument("--output_dir", default="./nsga2_results",
# #                         help="Directory to save results")
# #     parser.add_argument("--seed", type=int, default=42,
# #                         help="Random seed")
    
# #     args = parser.parse_args()
    
# #     # Run optimization
# #     result = run_vgg11_nsga2_optimization(
# #         dataset=args.dataset,
# #         data_root=args.data_root,
# #         weights_path=args.weights_path,
# #         img_size=args.img_size,
# #         population=args.population,
# #         generations=args.generations,
# #         eval_samples=args.eval_samples,
# #         batch_size=args.batch_size,
# #         mutation_rate=args.mutation_rate,
# #         posit_conv_indices=args.posit_conv_indices,
# #         output_dir=args.output_dir,
# #         seed=args.seed
# #     )
    
# #     if result is None:
# #         log("Optimization failed!")
# #         exit(1)
    
# #     log("Optimization completed successfully!")

# import numpy as np
# import torch
# import random
# import csv
# import os
# import argparse
# from datetime import datetime

# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

# # Import the modified VGG11 model and evaluation functions
# from vgg11_accuracy import VGG11, evaluate_vgg11_with_posit, get_test_loader, log

# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.termination import get_termination
# from pymoo.optimize import minimize
# from pymoo.operators.sampling.rnd import IntegerRandomSampling
# from pymoo.operators.crossover.pntx import PointCrossover
# from pymoo.core.mutation import Mutation

# # =========================================================
# # Global Config
# # =========================================================
# POPULATION = 50
# GENERATIONS = 20
# SEED = 42
# EVAL_SAMPLES = 1000          # how many images to evaluate per candidate
# BATCH_SIZE = 32
# MUTATION_RATE = 0.30

# # Allowed posit pairs (index -> (N, es))
# VALID_PAIRS = [(6, 2), (8, 2), (7, 2), (9, 2)]

# # Dataset and model configuration
# DEFAULT_DATASET = "cifar10"
# DEFAULT_DATA_ROOT = "./data"
# DEFAULT_WEIGHTS_PATH = "./models/vgg11_cifar10.pth"

# # Which convs run with posit arithmetic (by index in VGG11 forward)
# # 0=conv1, 1=conv2, 2=conv3, 3=conv4, 4=conv5, 5=conv6, 6=conv7, 7=conv8
# POSIT_CONV_INDICES = [0]  # Start with first conv only

# # =========================================================
# # Hardware Cost Model (simple)
# # =========================================================
# def hardware_cost_graded(config):
#     """
#     Cost proxy: sum of N^2 per-PE, plus 10% coordination overhead.
#     config: list of 9 tuples [(N, es), ...]
#     """
#     total = sum(N * N for (N, es) in config)
#     return total * 1.10

# def generation_callback(algorithm):
#     """Callback function to track generation progress"""
#     print(f"Generation {algorithm.n_gen} completed")

# # =========================================================
# # Custom Mutation (encourages local tweaks + occasional global jumps)
# # =========================================================
# class GradedPrecisionMutation(Mutation):
#     def __init__(self, prob=0.3):
#         super().__init__()
#         self.prob = prob

#     def _do(self, problem, X, **kwargs):
#         X = X.astype(int)
#         n_choices = len(VALID_PAIRS)
#         for i in range(len(X)):
#             # per-PE edits
#             for j in range(X.shape[1]):
#                 if np.random.rand() < self.prob:
#                     if np.random.rand() < 0.7:  # local tweak
#                         delta = np.random.choice([-1, 0, 1])
#                         X[i, j] = int(np.clip(X[i, j] + delta, 0, n_choices - 1))
#                     else:  # global jump
#                         X[i, j] = np.random.randint(0, n_choices)
#         return X

# # =========================================================
# # NSGA-II Problem (9 vars = 3x3 PEs)
# # =========================================================
# class VGG11GradedPEProblem(ElementwiseProblem):
#     def __init__(self, model, dataloader, device, log_file):
#         super().__init__(
#             n_var=9,                 # 9 PEs for a 3x3 kernel
#             n_obj=2,                 # minimize [-accuracy, cost]
#             xl=0, xu=len(VALID_PAIRS) - 1,
#             type_var=int
#         )
#         self.model = model
#         self.loader = dataloader
#         self.device = device
#         self.log_file = log_file
#         self.evals = 0

#     def _evaluate(self, x, out, *args, **kwargs):
#         # map indices -> actual (N, es)
#         pe_config = [VALID_PAIRS[int(i)] for i in x]

#         # Debug: Remove print from here since it can cause issues
#         # run VGG11 evaluation with posit arithmetic
#         correct, total = evaluate_vgg11_with_posit(
#             self.model, self.loader,
#             POSIT_CONV_INDICES, pe_config,
#             num_samples=EVAL_SAMPLES, device=self.device
#         )
#         acc = 100.0 * correct / total if total > 0 else 0.0
#         cost = hardware_cost_graded(pe_config)

#         # NSGA-II minimizes objectives → negate accuracy
#         out["F"] = [-acc, cost]

#         # log each evaluation
#         with open(self.log_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", str(pe_config)])

#         self.evals += 1
#         log(f"Evaluation {self.evals}: Accuracy={acc:.2f}%, Cost={cost:.2f}")

# # =========================================================
# # Dataset configuration mapping
# # =========================================================
# def get_dataset_config(dataset_name):
#     """Get dataset-specific configuration"""
#     ds = dataset_name.lower()
    
#     if ds == "cifar10":
#         return {
#             'mean': (0.4914, 0.4822, 0.4465),
#             'std': (0.2023, 0.1994, 0.2010),
#             'num_classes': 10,
#             'img_size': 32
#         }
#     elif ds == "cifar100":
#         return {
#             'mean': (0.4914, 0.4822, 0.4465),
#             'std': (0.2023, 0.1994, 0.2010),
#             'num_classes': 100,
#             'img_size': 32
#         }
#     elif ds == "svhn":
#         return {
#             'mean': (0.4377, 0.4438, 0.4728),
#             'std': (0.198, 0.201, 0.197),
#             'num_classes': 10,
#             'img_size': 32
#         }
#     elif ds in ("imagewoof", "folder"):
#         return {
#             'mean': (0.485, 0.456, 0.406),
#             'std': (0.229, 0.224, 0.225),
#             'num_classes': None,  # Will be determined from dataset
#             'img_size': 224
#         }
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

# # =========================================================
# # Main optimization function
# # =========================================================
# def run_vgg11_nsga2_optimization(
#     dataset="cifar10",
#     data_root="./data",
#     weights_path="./models/vgg11.pth",
#     img_size=None,
#     population=50,
#     generations=20,
#     eval_samples=1000,
#     batch_size=32,
#     mutation_rate=0.3,
#     posit_conv_indices=None,
#     output_dir="./nsga2_results",
#     seed=42
# ):
#     """Run NSGA-II optimization for VGG11 posit precision"""
    
#     # Setup
#     if posit_conv_indices is None:
#         posit_conv_indices = [0]
    
#     global POSIT_CONV_INDICES, EVAL_SAMPLES, BATCH_SIZE
#     POSIT_CONV_INDICES = posit_conv_indices
#     EVAL_SAMPLES = eval_samples
#     BATCH_SIZE = batch_size
    
#     # Set seeds
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     device = "cpu"
#     log(f"Using device: {device}")

#     # Get dataset configuration
#     dataset_config = get_dataset_config(dataset)
#     if img_size is None:
#         img_size = dataset_config['img_size']

#     # Load dataset
#     try:
#         test_loader, num_classes = get_test_loader(
#             dataset, data_root, batch_size, img_size=img_size, download=True
#         )
#         log(f"Dataset: {dataset} | Classes: {num_classes}")
#     except Exception as e:
#         log(f"Error loading dataset: {e}")
#         return None

#     # Load model
#     model = VGG11(num_classes=num_classes).to(device)
#     try:
#         state = torch.load(weights_path, map_location=device)
#         model.load_state_dict(state)
#         log("Model loaded successfully")
#     except FileNotFoundError:
#         log(f"Model file {weights_path} not found. Using random weights.")
#     except Exception as e:
#         log(f"Error loading model: {e}")
#         return None

#     model.eval()

#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Setup logging
#     log_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_log.csv")
#     with open(log_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["time", "accuracy", "hardware_cost", "pe_config(9)"])

#     # Setup NSGA-II problem
#     log("Creating optimization problem...")
#     problem = VGG11GradedPEProblem(model, test_loader, device, log_file)
#     log("Problem created successfully")
    
#     algorithm = NSGA2(
#         pop_size=population,
#         sampling=IntegerRandomSampling(),
#         crossover=PointCrossover(n_points=2),
#         mutation=GradedPrecisionMutation(prob=mutation_rate),
#         eliminate_duplicates=True
#     )

#     log(f"Starting NSGA-II optimization:")
#     log(f"  Population: {population}")
#     log(f"  Generations: {generations}")
#     log(f"  Evaluation samples: {eval_samples}")
#     log(f"  Posit conv indices: {posit_conv_indices}")
#     log(f"  Valid posit pairs: {VALID_PAIRS}")

#     # Run optimization
#     res = minimize(
#         problem, algorithm,
#         termination=get_termination("n_gen", generations),
#         seed=seed, verbose=True,
#         callback=generation_callback
#     )

#     # Save and display results
#     results_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_pareto.csv")
#     with open(results_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["accuracy", "hardware_cost", "pe_config"])
        
#         log("\n" + "="*60)
#         log("PARETO FRONT RESULTS:")
#         log("="*60)
        
#         for i, (x, f) in enumerate(zip(res.X, res.F)):
#             cfg = [VALID_PAIRS[int(j)] for j in x]
#             acc = -f[0]  # Convert back from minimization
#             cost = f[1]
            
#             writer.writerow([f"{acc:.2f}", f"{cost:.2f}", str(cfg)])
#             log(f"Solution {i+1:2d}: Acc={acc:6.2f}%  Cost={cost:8.2f}  Config={cfg}")

#     log(f"\nOptimization completed!")
#     log(f"Results saved to: {results_file}")
#     log(f"Full log saved to: {log_file}")
    
#     return res

# # =========================================================
# # Command Line Interface
# # =========================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("VGG11 NSGA-II Posit Precision Optimization")
    
#     # Dataset arguments
#     parser.add_argument("--dataset", default="cifar10",
#                         choices=["cifar10", "cifar100", "svhn", "imagewoof", "folder"],
#                         help="Dataset to use for optimization")
#     parser.add_argument("--data_root", default="./data",
#                         help="Root directory for dataset")
#     parser.add_argument("--weights_path", default="./models/vgg11.pth",
#                         help="Path to trained VGG11 model weights")
#     parser.add_argument("--img_size", type=int, default=None,
#                         help="Image size (default: 32 for CIFAR/SVHN, 224 for ImageFolder)")
    
#     # Optimization arguments
#     parser.add_argument("--population", type=int, default=50,
#                         help="NSGA-II population size")
#     parser.add_argument("--generations", type=int, default=20,
#                         help="Number of generations")
#     parser.add_argument("--eval_samples", type=int, default=1000,
#                         help="Number of samples to evaluate per candidate")
#     parser.add_argument("--batch_size", type=int, default=32,
#                         help="Batch size for evaluation")
#     parser.add_argument("--mutation_rate", type=float, default=0.3,
#                         help="Mutation probability")
    
#     # Posit configuration
#     parser.add_argument("--posit_conv_indices", nargs="+", type=int, default=[0],
#                         help="Indices of conv layers to use posit arithmetic (0-7)")
    
#     # Output and misc
#     parser.add_argument("--output_dir", default="./nsga2_results",
#                         help="Directory to save results")
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed")
    
#     args = parser.parse_args()
    
#     # Run optimization
#     result = run_vgg11_nsga2_optimization(
#         dataset=args.dataset,
#         data_root=args.data_root,
#         weights_path=args.weights_path,
#         img_size=args.img_size,
#         population=args.population,
#         generations=args.generations,
#         eval_samples=args.eval_samples,
#         batch_size=args.batch_size,
#         mutation_rate=args.mutation_rate,
#         posit_conv_indices=args.posit_conv_indices,
#         output_dir=args.output_dir,
#         seed=args.seed
#     )
    
#     if result is None:
#         log("Optimization failed!")
#         exit(1)
    
#     log("Optimization completed successfully!")

import numpy as np
import torch
import random
import csv
import os
import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

# Import the modified VGG11 model and evaluation functions
from vgg11_accuracy import VGG11, evaluate_vgg11_with_posit, get_test_loader, log

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.core.mutation import Mutation

# =========================================================
# Global Config
# =========================================================
POPULATION = 1
GENERATIONS = 1
SEED = 42
EVAL_SAMPLES = 1000          # how many images to evaluate per candidate
BATCH_SIZE = 32
MUTATION_RATE = 0.30

# Allowed posit pairs (index -> (N, es))
VALID_PAIRS = [(8, 2), (9, 2), (10, 2)]

# Dataset and model configuration
DEFAULT_DATASET = "cifar10"
DEFAULT_DATA_ROOT = "./data"
DEFAULT_WEIGHTS_PATH = "./models/vgg11_cifar10.pth"

# Which convs run with posit arithmetic (by index in VGG11 forward)
# 0=conv1, 1=conv2, 2=conv3, 3=conv4, 4=conv5, 5=conv6, 6=conv7, 7=conv8
POSIT_CONV_INDICES = [0]  # Start with first conv only

# =========================================================
# Hardware Cost Model (simple)
# =========================================================
def hardware_cost_graded(config):
    """
    Cost proxy: sum of N^2 per-PE, plus 10% coordination overhead.
    config: list of 9 tuples [(N, es), ...]
    """
    total = sum(N * N for (N, es) in config)
    return total * 1.10

def generation_callback(algorithm):
    """Callback function to track generation progress"""
    print(f"Generation {algorithm.n_gen} completed")

# =========================================================
# Custom Mutation (encourages local tweaks + occasional global jumps)
# =========================================================
class GradedPrecisionMutation(Mutation):
    def __init__(self, prob=0.3):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X = X.astype(int)
        n_choices = len(VALID_PAIRS)
        for i in range(len(X)):
            # per-PE edits
            for j in range(X.shape[1]):
                if np.random.rand() < self.prob:
                    if np.random.rand() < 0.7:  # local tweak
                        delta = np.random.choice([-1, 0, 1])
                        X[i, j] = int(np.clip(X[i, j] + delta, 0, n_choices - 1))
                    else:  # global jump
                        X[i, j] = np.random.randint(0, n_choices)
        return X

# =========================================================
# NSGA-II Problem (9 vars = 3x3 PEs)
# =========================================================
class VGG11GradedPEProblem(ElementwiseProblem):
    def __init__(self, model, dataloader, device, log_file):
        super().__init__(
            n_var=9,                 # 9 PEs for a 3x3 kernel
            n_obj=2,                 # minimize [-accuracy, cost]
            xl=0, xu=len(VALID_PAIRS) - 1,
            type_var=int
        )
        self.model = model
        self.loader = dataloader
        self.device = device
        self.log_file = log_file
        self.evals = 0

    # def _evaluate(self, x, out, *args, **kwargs):
    #     # map indices -> actual (N, es)
    #     pe_config = [VALID_PAIRS[int(i)] for i in x]

    #     # run VGG11 evaluation with posit arithmetic
    #     correct, total = evaluate_vgg11_with_posit(
    #         self.model, self.loader,
    #         POSIT_CONV_INDICES, pe_config,
    #         num_samples=EVAL_SAMPLES, device=self.device
    #     )
    #     acc = 100.0 * correct / total if total > 0 else 0.0
    #     cost = hardware_cost_graded(pe_config)

    #     # NSGA-II minimizes objectives → negate accuracy
    #     out["F"] = [-acc, cost]

    #     # log each evaluation
    #     with open(self.log_file, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", str(pe_config)])

    #     self.evals += 1
    #     log(f"Evaluation {self.evals}: Accuracy={acc:.2f}%, Cost={cost:.2f}")

    def _evaluate(self, x, out, *args, **kwargs):
        import copy
        
        # Create fresh model copy to avoid state corruption
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        
        # Convert integer indices to PE config tuples
        pe_config = [VALID_PAIRS[int(i)] for i in x]
        
        # Evaluate with isolated model
        correct, total = evaluate_vgg11_with_posit(
            model_copy,
            self.loader,
            POSIT_CONV_INDICES,
            pe_config,
            EVAL_SAMPLES,
            self.device
        )
        
        acc = 100.0 * correct / total if total > 0 else 0.0
        cost = hardware_cost_graded(pe_config)
        
        out["F"] = [-acc, cost]
        
        # Log result
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                f"{acc:.3f}",
                f"{cost:.2f}",
                str(pe_config)
            ])
        
        self.evals += 1
        log(f"Evaluation {self.evals}: Accuracy={acc:.2f}%, Cost={cost:.2f}")


# =========================================================
# Dataset configuration mapping
# =========================================================
def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    ds = dataset_name.lower()
    
    if ds == "cifar10":
        return {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'num_classes': 10,
            'img_size': 32
        }
    elif ds == "cifar100":
        return {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'num_classes': 100,
            'img_size': 32
        }
    elif ds == "svhn":
        return {
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.198, 0.201, 0.197),
            'num_classes': 10,
            'img_size': 32
        }
    elif ds in ("imagewoof", "folder"):
        return {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'num_classes': None,  # Will be determined from dataset
            'img_size': 224
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# =========================================================
# Main optimization function
# =========================================================
def run_vgg11_nsga2_optimization(
    dataset="cifar10",
    data_root="./data",
    weights_path="./models/vgg11.pth",
    img_size=None,
    population=50,
    generations=20,
    eval_samples=1000,
    batch_size=32,
    mutation_rate=0.3,
    posit_conv_indices=None,
    output_dir="./nsga2_results",
    seed=42
):
    """Run NSGA-II optimization for VGG11 posit precision"""
    
    # Setup
    if posit_conv_indices is None:
        posit_conv_indices = [0]
    
    global POSIT_CONV_INDICES, EVAL_SAMPLES, BATCH_SIZE
    POSIT_CONV_INDICES = posit_conv_indices
    EVAL_SAMPLES = eval_samples
    BATCH_SIZE = batch_size
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cpu"
    log(f"Using device: {device}")

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset)
    if img_size is None:
        img_size = dataset_config['img_size']

    # Load dataset
    try:
        test_loader, num_classes = get_test_loader(
            dataset, data_root, batch_size, img_size=img_size, download=True
        )
        log(f"Dataset: {dataset} | Classes: {num_classes}")
    except Exception as e:
        log(f"Error loading dataset: {e}")
        return None

    # Load model
    model = VGG11(num_classes=num_classes).to(device)
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        log("Model loaded successfully")
    except FileNotFoundError:
        log(f"Model file {weights_path} not found. Using random weights.")
    except Exception as e:
        log(f"Error loading model: {e}")
        return None

    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "accuracy", "hardware_cost", "pe_config(9)"])

    # Setup NSGA-II problem
    log("Creating optimization problem...")
    torch.set_num_threads(1)
    problem = VGG11GradedPEProblem(model, test_loader, device, log_file)
    log("Problem created successfully")
    
    algorithm = NSGA2(
        pop_size=population,
        sampling=IntegerRandomSampling(),
        crossover=PointCrossover(n_points=2),
        mutation=GradedPrecisionMutation(prob=mutation_rate),
        eliminate_duplicates=True
    )

    log(f"Starting NSGA-II optimization:")
    log(f"  Population: {population}")
    log(f"  Generations: {generations}")
    log(f"  Evaluation samples: {eval_samples}")
    log(f"  Posit conv indices: {posit_conv_indices}")
    log(f"  Valid posit pairs: {VALID_PAIRS}")

    # Run optimization
    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", generations),
        seed=seed, verbose=True,
        callback=generation_callback
    )

    # Save and display results
    results_file = os.path.join(output_dir, f"nsga2_vgg11_{dataset}_pareto.csv")
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "hardware_cost", "pe_config"])
        
        log("\n" + "="*60)
        log("PARETO FRONT RESULTS:")
        log("="*60)
        
        for i, (x, f) in enumerate(zip(res.X, res.F)):
            cfg = [VALID_PAIRS[int(j)] for j in x]
            acc = -f[0]  # Convert back from minimization
            cost = f[1]
            
            writer.writerow([f"{acc:.2f}", f"{cost:.2f}", str(cfg)])
            log(f"Solution {i+1:2d}: Acc={acc:6.2f}%  Cost={cost:8.2f}  Config={cfg}")

    log(f"\nOptimization completed!")
    log(f"Results saved to: {results_file}")
    log(f"Full log saved to: {log_file}")
    
    return res

# =========================================================
# Command Line Interface
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("VGG11 NSGA-II Posit Precision Optimization")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "svhn", "imagewoof", "folder"],
                        help="Dataset to use for optimization")
    parser.add_argument("--data_root", default="./data",
                        help="Root directory for dataset")
    parser.add_argument("--weights_path", default="./models/vgg11.pth",
                        help="Path to trained VGG11 model weights")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Image size (default: 32 for CIFAR/SVHN, 224 for ImageFolder)")
    
    # Optimization arguments
    parser.add_argument("--population", type=int, default=50,
                        help="NSGA-II population size")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of generations")
    parser.add_argument("--eval_samples", type=int, default=1000,
                        help="Number of samples to evaluate per candidate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--mutation_rate", type=float, default=0.3,
                        help="Mutation probability")
    
    # Posit configuration
    parser.add_argument("--posit_conv_indices", nargs="+", type=int, default=[0],
                        help="Indices of conv layers to use posit arithmetic (0-7)")
    
    # Output and misc
    parser.add_argument("--output_dir", default="./nsga2_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Run optimization
    result = run_vgg11_nsga2_optimization(
        dataset=args.dataset,
        data_root=args.data_root,
        weights_path=args.weights_path,
        img_size=args.img_size,
        population=args.population,
        generations=args.generations,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        mutation_rate=args.mutation_rate,
        posit_conv_indices=args.posit_conv_indices,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if result is None:
        log("Optimization failed!")
        exit(1)
    
    log("Optimization completed successfully!")
