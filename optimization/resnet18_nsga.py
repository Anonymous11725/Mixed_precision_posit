import numpy as np
import torch
import random
import csv
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# --- bring in your model and evaluator from resnet18_accuracy.py ---
from resnet18_accuracy import ResNet18, evaluate_resnet18_with_posit, log

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
POPULATION = 50
GENERATIONS = 20
SEED = 42
EVAL_SAMPLES = 1000          # how many images to evaluate per candidate
BATCH_SIZE = 32
MUTATION_RATE = 0.30

# Allowed posit pairs (index -> (N, es))
VALID_PAIRS = [(6, 2), (8, 2), (7, 2), (9, 2)]

# Choose dataset and weights
DATASET = "SVHN"        # "CIFAR10" | "CIFAR100" | "SVHN"
WEIGHTS_PATH = "/home/sneha/sneha/bird/booth_systolic/python_codes/new_inferences/models/resnet18_svhn.pth"

# Which convs run with posit arithmetic (by index in your forward) – start with first conv only
POSIT_CONV_INDICES = [0]

LOG_FILE = "nsga2_resnet18_graded_log_cifar100.csv"

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
class ResNet18GradedPEProblem(ElementwiseProblem):
    def __init__(self, model, dataloader, device):
        super().__init__(
            n_var=9,                 # 9 PEs for a 3x3 kernel
            n_obj=2,                 # minimize [-accuracy, cost]
            xl=0, xu=len(VALID_PAIRS) - 1,
            type_var=int
        )
        self.model = model
        self.loader = dataloader
        self.device = device
        self.evals = 0

    def _evaluate(self, x, out, *args, **kwargs):
        # map indices -> actual (N, es)
        pe_config = [VALID_PAIRS[int(i)] for i in x]

        # run your existing evaluator (returns correct, total)
        correct, total = evaluate_resnet18_with_posit(
            self.model, self.loader,
            POSIT_CONV_INDICES, pe_config,
            num_samples=EVAL_SAMPLES, device=self.device
        )
        acc = 100.0 * correct / total if total > 0 else 0.0
        cost = hardware_cost_graded(pe_config)

        # NSGA-II minimizes objectives → negate accuracy
        out["F"] = [-acc, cost]

        # log each evaluation
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), f"{acc:.3f}", f"{cost:.2f}", pe_config])

        self.evals += 1


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # seeds
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # -------- dataset + transforms (match typical CIFAR/SVHN norms) --------
    if DATASET == "CIFAR10":
        MEAN, STD, NUM_CLASSES = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10
        ds = CIFAR10(root="./data", train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(MEAN, STD)]))
    elif DATASET == "CIFAR100":
        MEAN, STD, NUM_CLASSES = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 100
        ds = CIFAR100(root="/home/sneha/sneha/bird/posit_sys/python_codes/data/cifar100/", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(MEAN, STD)]))
    else:  # SVHN
        MEAN, STD, NUM_CLASSES = (0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197), 10
        ds = SVHN(root="/home/sneha/sneha/bird/booth_systolic/python_codes/data", split="test", download=True,
                  transform=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(MEAN, STD)]))

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # -------- model --------
    model = ResNet18(num_classes=NUM_CLASSES).to(device)
    try:
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        log("Model loaded successfully")
    except FileNotFoundError:
        log(f"Model file {WEIGHTS_PATH} not found. Using random weights.")

    model.eval()

    # -------- CSV header --------
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "accuracy", "hardware_cost", "pe_config(9)"])

    # -------- NSGA-II --------
    problem = ResNet18GradedPEProblem(model, loader, device)
    algorithm = NSGA2(
        pop_size=POPULATION,
        sampling=IntegerRandomSampling(),           # random integer init in [xl, xu]
        crossover=PointCrossover(n_points=2),
        mutation=GradedPrecisionMutation(prob=MUTATION_RATE),
        eliminate_duplicates=True
    )

    log(f"Starting NSGA-II: pop={POPULATION}, gens={GENERATIONS}, eval_samples={EVAL_SAMPLES}")
    res = minimize(
        problem, algorithm,
        termination=get_termination("n_gen", GENERATIONS),
        seed=SEED, verbose=True,
        callback=generation_callback
    )

    # -------- Print Pareto --------
    print("\nPareto Front:")
    for x, f in zip(res.X, res.F):
        cfg = [VALID_PAIRS[int(i)] for i in x]
        acc = -f[0]
        cost = f[1]
        print(f"Acc={acc:6.2f}%  Cost={cost:8.2f}  Config={cfg}")
