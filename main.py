# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from train import main as train_main
import os
from datetime import datetime

def compare_methods(dev_mode=True):
    """
    CNN+Attention, Transformer+Contrastive, Multi-Branch CNN ve CNN+LSTM yöntemlerini karşılaştır
    
    Args:
        dev_mode (bool): Geliştirme modu için True, tam eğitim için False
    """
    print("Starting comparative experiment of multiple HAR methods...")
    
    # Tüm yöntemleri çalıştır
    _, cnn_accuracy = train_main(dev_mode=dev_mode, method="cnn_attention")
    _, transformer_accuracy = train_main(dev_mode=dev_mode, method="transformer_contrastive")
    _, multi_branch_accuracy = train_main(dev_mode=dev_mode, method="multi_branch_cnn")
    _, cnn_lstm_accuracy = train_main(dev_mode=dev_mode, method="cnn_lstm")
    
    # Karşılaştırma sonuçlarını kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join("output", f"comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Karşılaştırma grafiği
    methods = ['CNN+Attention', 'Transformer+Contrastive', 'Multi-Branch CNN', 'CNN+LSTM']
    accuracies = [cnn_accuracy, transformer_accuracy, multi_branch_accuracy, cnn_lstm_accuracy]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(methods, accuracies, color=['blue', 'orange', 'green', 'red'])
    plt.title('Method Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Çubukların üzerine değerleri yaz
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(comparison_dir, 'method_comparison.png'))
    plt.close()
    
    # Sonuçları metin dosyasına kaydet
    with open(os.path.join(comparison_dir, 'comparison_results.txt'), 'w') as f:
        f.write(f"Comparison Results (dev_mode={dev_mode}):\n\n")
        f.write(f"CNN+Attention Accuracy: {cnn_accuracy:.2f}%\n")
        f.write(f"Transformer+Contrastive Accuracy: {transformer_accuracy:.2f}%\n")
        f.write(f"Multi-Branch CNN Accuracy: {multi_branch_accuracy:.2f}%\n")
        f.write(f"CNN+LSTM Accuracy: {cnn_lstm_accuracy:.2f}%\n\n")
        
        # En iyi yöntemi bul
        best_idx = np.argmax(accuracies)
        best_method = methods[best_idx]
        f.write(f"Best Method: {best_method} ({accuracies[best_idx]:.2f}%)")
    
    print("\nComparative experiment completed!")
    print(f"Results saved to: {comparison_dir}")
    print(f"CNN+Attention Accuracy: {cnn_accuracy:.2f}%")
    print(f"Transformer+Contrastive Accuracy: {transformer_accuracy:.2f}%")
    print(f"Multi-Branch CNN Accuracy: {multi_branch_accuracy:.2f}%")
    print(f"CNN+LSTM Accuracy: {cnn_lstm_accuracy:.2f}%")
    print(f"Best Method: {methods[np.argmax(accuracies)]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human Activity Recognition with Multiple Methods')
    parser.add_argument('--dev', action='store_true', help='Run in development mode with less data')
    parser.add_argument('--method', type=str, 
                        choices=['cnn_attention', 'transformer_contrastive', 'multi_branch_cnn', 'cnn_lstm', 'compare'], 
                        default='cnn_attention', help='Which method to use')
    
    args = parser.parse_args()
    
    if args.method == 'compare':
        compare_methods(dev_mode=args.dev)
    else:
        train_main(dev_mode=args.dev, method=args.method)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
