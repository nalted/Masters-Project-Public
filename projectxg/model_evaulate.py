import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc, confusion_matrix
from model_train import load_data, process_features

# loads a trained model from a saved file
def load_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def plot_roc_curve(Y_true, Y_pred, output_path=None):
    fpr, tpr, _ = roc_curve(Y_true, Y_pred) # false positive rate, true positive rate
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(tpr, 1 - fpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('signal efficiency', fontsize=12)
    plt.ylabel('background rejection', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {output_path}")
    else:
        plt.show()
    plt.close()
    # to save in pictures/ folder, python model_evaluate.py --model-in ....json --input-files ....root --output-prefix pictures/my_model_eval

def plot_score_distribution(Y_true, Y_pred, output_path=None):
    signal_scores = Y_pred[Y_true == 1]
    background_scores = Y_pred[Y_true == 0]
    
    plt.figure(figsize=(8, 6))
    plt.hist(signal_scores, bins=50, alpha=0.7, label='Signal (electrons)', density=True, color='blue')
    plt.hist(background_scores, bins=50, alpha=0.7, label='Background', density=True, color='red')
    plt.xlabel('Model Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution', fontsize=14)
    plt.legend(loc='upper center', fontsize=11)
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved score distribution to {output_path}")
    else:
        plt.show()
    plt.close()

def evaluate_model(model, X, Y):
    # Prepare data for prediction
    X_array = np.asarray(X[["E_over_p", "isolation_frac", "is_leading", "charge"]])
    dmatrix = xgb.DMatrix(X_array)
    
    # Get predictions
    Y_pred = model.predict(dmatrix)
    
    return Y_pred

def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost electron ID model")
    parser.add_argument('--model-in', type=str, required=True,
                       help='Path to trained model file (e.g., xgb_18x275_28jan2026.json)')
    parser.add_argument('--input-files', type=str, nargs='+', required=True,
                       help='Input ROOT files for evaluation')
    parser.add_argument('--output-prefix', type=str, default='eval',
                       help='Prefix for output plot files (default: eval)')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_in}...")
    model = load_model(args.model_in)
    
    print(f"Loading evaluation data from {len(args.input_files)} files...")
    events = load_data(args.input_files)
    
    print("Processing features...")
    X_df, Y_df = process_features(events)
    
    print(f"Evaluating model on {len(X_df)} samples...")
    Y_pred = evaluate_model(model, X_df, Y_df)
    
    print("Generating plots...")
    plot_roc_curve(Y_df, Y_pred, output_path=f"{args.output_prefix}_roc.png")
    plot_score_distribution(Y_df, Y_pred, output_path=f"{args.output_prefix}_scores.png")
    
    print("Done!")

if __name__ == '__main__':
    main()