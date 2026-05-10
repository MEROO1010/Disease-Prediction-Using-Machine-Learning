import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot a confusion matrix.
    Args:
        cm: Confusion matrix (numpy array).
        class_names: List of class names.
        save_path: Path to save the plot (optional).
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_results(results, file_path):
    """
    Save evaluation results to a markdown file.
    Args:
        results: Dictionary of results.
        file_path: Path to save the results.
    """
    with open(file_path, 'w') as f:
        f.write("# Model Performance Report\n\n")
        for name, result in results.items():
            f.write(f"## {name}\n")
            f.write(f"- Accuracy: {result['accuracy']:.4f}\n")
            f.write("- Classification Report:\n")
            for key, value in result['classification_report'].items():
                f.write(f"  - {key}: {value}\n")
            f.write("\n")