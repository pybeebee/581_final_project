### IMPORT STATEMENTS

from itertools import product
import os, argparse, json

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed = 1


### PARSE ARGUMENTS

def parse_args():
    """
    Define and parse script arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--python_path", 
        type=str, 
        default="./", 
        help="What the default Python path should be set to"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="emotion", 
        help="Whether to work on emotion or intensity classification"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="mfcc", 
        help="Whether to use mfcc or gfcc features"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="svm", 
        help="Whether to run training for SVM, logreg, perceptron, or MLP models"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results", 
        help="Where to save the classification report and confusion matrix results"
    )
    
    args = parser.parse_args()
    return args


### HELPER FUNCTIONS

def score_model(targets, preds, output_dir, save_str, split):
    """
    Given target and predicted labels, generate classification report to obtain accuracy, precision, recall, F1 for each class and overall.
    """

    # Obtain class labels
    if 'emotion' in output_dir:
        classes = LABEL_MAP['emotion']
    else:
        classes = LABEL_MAP['intensity']
        
    # Flatten inputs 
    targets = targets.flatten()
    preds = preds.flatten()
    
    # Compute classification report
    report_dict = classification_report(targets, preds, target_names=classes, output_dict=True)

    # Save results
    with open(f"{output_dir}/cr__{save_str}__{split}.json", "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to {output_dir}/cr__{save_str}__{split}.json")


def save_cm(targets, preds, output_dir, save_str, plot_title, split):
    """
    Given target and predicted labels, generate confusion matrix.
    """

    # Obtain class labels
    if 'emotion' in output_dir:
        classes = LABEL_MAP['emotion']
    else:
        classes = LABEL_MAP['intensity']
        
    # Flatten inputs 
    targets = targets.flatten()
    preds = preds.flatten()

    # Generate confusion matrix & save plot
    cm = confusion_matrix(targets, preds, normalize='true') # normalize over true values
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    cm_display.plot(cmap="BuGn", values_format='.2f')
    plt.xticks(plt.xticks()[0], classes, rotation='vertical')
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm__{save_str}__{split}.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_dir}/cm__{save_str}__{split}.png")


### TRAINING LOOP FUNCTIONS

def train_svm(output_dir, data):
    """
    Run SVM model training with various hyperparameters.
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    x_train, y_train, x_val, y_val, x_test, y_test = data 


    # Define Model Parameters
    kernels = [
        'linear',
        'rbf',
        'sigmoid',
    ]
    degrees = [2,3,4,5] # skip degree 1 since that's just linear
    random_states = [1,2,3,4,5] # not really needed but keep for consistency in analysis scripts

    # Train SVM Model
    kernel_params = [(k, {}) for k in kernels]
    poly_params = [('poly', {'degree': d}) for d in degrees]
    all_params = kernel_params + poly_params

    for (kernel, extra_params), random_state in product(all_params, random_states):

        degree = extra_params.get("degree", None)
        print("On SVM combination:", "kernel", kernel, "degree", degree, "rs", random_state)

        # Define & fit model
        model = svm.SVC(kernel=kernel, random_state=random_state, **extra_params)
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print("Training error occurred:", e)
            continue

        # Get predictions
        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)
        preds_test = model.predict(x_test)

        # Save scores
        save_str = f"svm__{kernel}"
        if degree: save_str += f"__deg_{degree}"
        save_str += f"__rs_{random_state}"

        score_model(y_train, preds_train, output_dir, save_str, split="train")
        score_model(y_val, preds_val, output_dir, save_str, split="val")
        score_model(y_test, preds_test, output_dir, save_str, split="test")

        # Save confusion matrix
        plot_title = f"SVM with {kernel.capitalize()} Kernel"
        if degree: plot_title += f" (d={degree}, seed={random_state})"
        else: plot_title += f" (seed={random_state})"

        save_cm(y_train, preds_train, output_dir, save_str, plot_title, split="train")
        save_cm(y_val, preds_val, output_dir, save_str, plot_title, split="val")
        save_cm(y_test, preds_test, output_dir, save_str, plot_title, split="test")

    print("Completed SVM training runs!")


def train_logreg(output_dir, data):
    """
    Run logistic regression classifier training with various hyperparameters.
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    """

    x_train, y_train, x_val, y_val, x_test, y_test = data 

    # Define Model Parameters
    iters = [
        100,
        500,
        1000,
        2000,
        5000,
        10000,
    ]
    lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    schedules = [
        'constant',
        'invscaling',
    ]
    random_states = [1,2,3,4,5]


    start=False
    # Train LogReg Model
    for max_iter, lr, schedule, random_state in product(iters, lrs, schedules, random_states):

        print("On LogReg combination:", "iters",max_iter, "lr", lr, "schedule", schedule, "rs", random_state)
        x = f"On LogReg combination: iters {max_iter} lr {lr} schedule {schedule} rs {random_state}"
        if (x=="On LogReg combination: iters 2000 lr 5e-06 schedule invscaling rs 3" and "mfcc_emotion" in output_dir) or (x=="On LogReg combination: iters 10000 lr 1e-05 schedule constant rs 2" and "gfcc_emotion" in output_dir):     # continue after running out of space on disk
            start = True 
        if start==False: 
            continue

        # Define & fit model
        model = SGDClassifier(penalty=None, alpha=0.0, loss="log_loss", fit_intercept=False, max_iter=max_iter, random_state=random_state, learning_rate=schedule, eta0=lr)     # assume intercept already fit as data is centered via standardization
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print("Training error occurred:", e)
            continue

        # Get predictions
        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)
        preds_test = model.predict(x_test)

        # Save scores
        save_str = f"logreg__iters_{max_iter}__lr_{lr}__sched_{schedule}__rs_{random_state}"

        score_model(y_train, preds_train, output_dir, save_str, split="train")
        score_model(y_val, preds_val, output_dir, save_str, split="val")
        score_model(y_test, preds_test, output_dir, save_str, split="test")

        # Save confusion matrix
        plot_title = f"Logistic Classifier Trained with {schedule.capitalize()} Schedule\n(max_iter={max_iter}, lr={lr}, seed={random_state})"

        save_cm(y_train, preds_train, output_dir, save_str, plot_title, split="train")
        save_cm(y_val, preds_val, output_dir, save_str, plot_title, split="val")
        save_cm(y_test, preds_test, output_dir, save_str, plot_title, split="test")

    print("Completed logistic classifier training runs!")


def train_perceptron(output_dir, data):
    """
    Run perceptron classifier training with various hyperparameters.
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron
    """

    x_train, y_train, x_val, y_val, x_test, y_test = data 

    # Define Model Parameters
    iters = [
        100,
        500,
        1000,
        2000,
        5000,
        10000,
    ]
    lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    random_states = [1,2,3,4,5]

    # Train Perceptron Model
    for max_iter, lr, random_state in product(iters, lrs, random_states):

        print("On Perceptron combination:", "iters", max_iter, "lr", lr, "rs", random_state)

        # Define & fit model
        model = SGDClassifier(penalty=None, alpha=0.0, loss="perceptron", fit_intercept=False, learning_rate="constant", max_iter=max_iter, random_state=random_state, eta0=lr)     # assume intercept already fit as data is centered via standardization
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print("Training error occurred:", e)
            continue

        # Get predictions
        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)
        preds_test = model.predict(x_test)

        # Save scores
        save_str = f"perceptron__iters_{max_iter}__lr_{lr}__rs_{random_state}"

        score_model(y_train, preds_train, output_dir, save_str, split="train")
        score_model(y_val, preds_val, output_dir, save_str, split="val")
        score_model(y_test, preds_test, output_dir, save_str, split="test")

        # Save confusion matrix
        plot_title = f"Perceptron Classifier (max_iter={max_iter}, lr={lr}, seed={random_state})"

        save_cm(y_train, preds_train, output_dir, save_str, plot_title, split="train")
        save_cm(y_val, preds_val, output_dir, save_str, plot_title, split="val")
        save_cm(y_test, preds_test, output_dir, save_str, plot_title, split="test")

    print("Completed perceptron training runs!")


def train_mlp(output_dir, data):
    """
    Run MLP classifier training with various hyperparameters.
    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier
    """

    x_train, y_train, x_val, y_val, x_test, y_test = data 

    # Define Model Parameters
    layer_sizes = [
        (100,),
        (10,),
        # (500,),
        # (1000,),
        # (100, 100),
        # (500, 500),
        # (1000, 1000),
    ]
    solvers = [
        'sgd',
        # 'adam'
    ]
    schedules = [
        'constant',
        'invscaling'
    ]
    iters = [
        100,
        500,
        1000,
        2000,
        # 5000,
    ]
    lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    random_states = [1,2,3,4,5]

    # Train MLP Model
    start = False
    for layer_size, solver, schedule, max_iter, lr, random_state in product(layer_sizes, solvers, schedules, iters, lrs, random_states):

        print("On MLP combination:", layer_size, solver, schedule, "iters", max_iter, "lr", lr, "rs", random_state)
        x = f"On MLP combination: {layer_size} {solver} {schedule} iters {max_iter} lr {lr} rs {random_state}"
        if (x=="On MLP combination: (100,) sgd constant iters 2000 lr 0.001 rs 3" and "mfcc_emotion" in output_dir) or (x=="On MLP combination: (100,) sgd constant iters 2000 lr 1e-05 rs 5" and "mfcc_intensity" in output_dir) or (x=="On MLP combination: (100,) sgd constant iters 1000 lr 5e-06 rs 2" and "gfcc_emotion" in output_dir) or (x=="On MLP combination: (100,) sgd constant iters 1000 lr 5e-06 rs 5" and "gfcc_intensity" in output_dir):       # continue after running out of space on disk
            start = True 
        if start==False: 
            continue

        # Define & fit model
        model = MLPClassifier(activation='relu', batch_size=int(x_train.shape[0]*0.1), hidden_layer_sizes=layer_size, solver=solver, learning_rate=schedule, max_iter=max_iter, random_state=random_state, learning_rate_init=lr)
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            print("Training error occurred:", e)
            continue

        # Get predictions
        preds_train = model.predict(x_train)
        preds_val = model.predict(x_val)
        preds_test = model.predict(x_test)

        # Save scores
        save_str = f"mlp__layers_{str(layer_size)}__{solver}__{schedule}__iters_{max_iter}__lr_{lr}__rs_{random_state}"

        score_model(y_train, preds_train, output_dir, save_str, split="train")
        score_model(y_val, preds_val, output_dir, save_str, split="val")
        score_model(y_test, preds_test, output_dir, save_str, split="test")

        # Save confusion matrix
        plot_title = f"MLP Classifier Trained with {schedule.capitalize()} Schedule\n(layers={layer_size}, max_iter={max_iter}, lr={lr}, seed={random_state})"

        save_cm(y_train, preds_train, output_dir, save_str, plot_title, split="train")
        save_cm(y_val, preds_val, output_dir, save_str, plot_title, split="val")
        save_cm(y_test, preds_test, output_dir, save_str, plot_title, split="test")

    print("Completed MLP training runs!")


### GLOBAL VARIABLES

TRAIN_FXN_MAP = {
    "svm": train_svm,
    "logreg": train_logreg,
    "perceptron": train_perceptron,
    "mlp": train_mlp,
}

LABEL_MAP = {
    "emotion": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
    "intensity": ["normal", "strong"],
}


### MAIN FUNCTION

def main(args):

    ### PREPARE DATA

    assert args.data in ['mfcc', 'gfcc']
    assert args.task in ['emotion', 'intensity']

    # Load data
    print("Loading data...")
    x = np.load(f'./data/{args.data}_{args.task}_x.npz')
    y = np.load(f'./data/{args.data}_{args.task}_y.npz')

    print("Processing data...")
    x = x.f.data
    y = y.f.data

    # Create train-val-test splits
    print("Splitting data...")
    shuffled_indices = np.random.permutation(x.shape[0])

    # Choose the first 75% as training set, next 10% as validation and the rest as testing
    train_split_idx = int(0.75 * x.shape[0])
    val_split_idx = int(0.85 * x.shape[0])

    train_indices = shuffled_indices[:train_split_idx]
    val_indices = shuffled_indices[train_split_idx:val_split_idx]
    test_indices = shuffled_indices[val_split_idx:]

    # Select the examples from x and y to construct our training, validation, testing sets
    x_train, y_train = x[train_indices, :], y[train_indices]
    x_val, y_val = x[val_indices, :], y[val_indices]
    x_test, y_test = x[test_indices, :], y[test_indices]

    # Standardize the data
    print("Scaling data...")
    scaler = StandardScaler()

    # Fit scaler to train set
    scaler.fit(x_train[:100])

    # Transform data
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    data = (x_train, y_train, x_val, y_val, x_test, y_test)

    output_dir = f"{args.output_dir}/{args.data}_{args.task}"
    os.makedirs(output_dir, exist_ok=True)

    ### RUN TRAINING
    
    # Train models as specified in arguments
    print("Performing hyperparameter search...")
    train_fxn = TRAIN_FXN_MAP[args.model]
    train_fxn(output_dir, data)



if __name__ == "__main__":
    
    
    args = parse_args()
    os.environ['PYTHONPATH'] = args.python_path
    main(args)
