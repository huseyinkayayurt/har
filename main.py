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
    
    # Karşılaştırma için ana klasörü oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join("output", f"comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Her model için alt klasörler oluştur
    methods = ['cnn_attention', 'transformer_contrastive', 'multi_branch_cnn', 'cnn_lstm']
    method_names = ['CNN+Attention', 'Transformer+Contrastive', 'Multi-Branch CNN', 'CNN+LSTM']
    
    results = []
    
    for method, method_name in zip(methods, method_names):
        print(f"\nTraining {method_name}...")
        
        # Her model için kendi klasörünü oluştur
        method_dir = os.path.join(comparison_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        
        # Model için alt klasörler
        models_dir = os.path.join(method_dir, "models")
        plots_dir = os.path.join(method_dir, "plots")
        logs_dir = os.path.join(method_dir, "logs")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Modeli eğit
        _, accuracy = train_main(
            dev_mode=dev_mode,
            method=method,
            is_comparison=True,
            comparison_dir=method_dir  # Her model için kendi klasörünü kullan
        )
        results.append(accuracy)
        
        print(f"{method_name} training completed. Accuracy: {accuracy:.2f}%")
    
    # Karşılaştırma sonuçları için klasör
    comparison_results_dir = os.path.join(comparison_dir, "comparison_results")
    os.makedirs(comparison_results_dir, exist_ok=True)
    
    # Karşılaştırma grafiği
    plt.figure(figsize=(14, 8))
    bars = plt.bar(method_names, results, color=['blue', 'orange', 'green', 'red'])
    plt.title('Method Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Çubukların üzerine değerleri yaz
    for i, acc in enumerate(results):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(comparison_results_dir, 'method_comparison.png'))
    plt.close()
    
    # Sonuçları metin dosyasına kaydet
    with open(os.path.join(comparison_results_dir, 'comparison_results.txt'), 'w') as f:
        f.write(f"Comparison Results (dev_mode={dev_mode}):\n\n")
        for method_name, accuracy in zip(method_names, results):
            f.write(f"{method_name} Accuracy: {accuracy:.2f}%\n")
        
        # En iyi yöntemi bul
        best_idx = np.argmax(results)
        best_method = method_names[best_idx]
        f.write(f"\nBest Method: {best_method} ({results[best_idx]:.2f}%)")
    
    print("\nComparative experiment completed!")
    print(f"Results saved to: {comparison_dir}")
    for method_name, accuracy in zip(method_names, results):
        print(f"{method_name} Accuracy: {accuracy:.2f}%")
    print(f"Best Method: {method_names[np.argmax(results)]}")

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
