---
sidebar_position: 9
title: "Exercício com Treinamento completo - MLP Profundo"
description: "Este exercício integra todas as técnicas abordadas no Módulo 3, criando um sistema completo de treinamento de MLPs profundos com PyTorch."
tags: [dropout, weight decay, batch normalization, mlp]
---

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xHlhGBJ0A-uU9mXB9fq5BvdXt_XSJxRG?usp=sharing)

# Exercício Prático Integrado: Treinamento Completo de Treinamento com MLP Profundo

Este exercício integra todas as técnicas abordadas no Módulo 3, criando um sistema completo de treinamento de redes neurais MLP profundas com PyTorch.

## 🎯 Objetivo

Implementar um sistema de treinamento robusto que inclui:
- Carregamento e preparação de dados para MLPs
- Tratamento de desbalanceamento
- Múltiplas técnicas de regularização
- Monitoramento e diagnóstico
- Checkpointing e inferência otimizada

## 📊 Dataset: CIFAR-10 Desbalanceado para MLP

Vamos criar uma versão artificialmente desbalanceada do CIFAR-10 e adaptá-la para treinar MLPs (imagens serão "achatadas").

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import copy
import time
from collections import defaultdict

# Configuração
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Configurações do experimento
CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'patience': 10,
    'min_delta': 0.001,
    'gradient_clip_norm': 1.0,
    'accumulation_steps': 2,
    'imbalance_classes': [0, 1, 2],  # Classes que serão reduzidas
    'imbalance_ratio': 0.1,  # Manter apenas 10% dessas classes
    'dropout_rate': 0.3,
    'weight_decay': 1e-4,
    'input_size': 32 * 32 * 3,  # CIFAR-10 achatado (3072 features)
    'hidden_sizes': [2048, 1024, 512, 256, 128, 64],  # Arquitetura profunda
    'num_classes': 10
}
```

## 🏗️ Arquitetura do MLP Profundo

```python
class DeepMLP(nn.Module):
    """
    MLP profundo com múltiplas técnicas de regularização
    """
    def __init__(self, input_size=3072, hidden_sizes=[2048, 1024, 512, 256, 128, 64], 
                 num_classes=10, dropout_rate=0.3, use_batch_norm=True):
        super(DeepMLP, self).__init__()
        
        self.input_size = input_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Construir camadas dinamicamente
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            # Camada linear
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Batch normalization (opcional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # Função de ativação
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
        
        # Camada de saída (sem ativação, dropout ou batch norm)
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        
        # Criar o modelo sequencial
        self.network = nn.Sequential(*layers)
        
        # Inicialização de pesos melhorada
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialização Xavier/Glorot para melhor convergência"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Achatar a entrada (batch_size, 3, 32, 32) -> (batch_size, 3072)
        x = x.view(x.size(0), -1)
        
        # Passar pela rede
        return self.network(x)
    
    def get_layer_info(self):
        """Retorna informações sobre as camadas da rede"""
        info = []
        layer_count = 0
        param_count = 0
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
                layer_params = sum(p.numel() for p in module.parameters())
                param_count += layer_params
                
                info.append({
                    'layer': layer_count,
                    'name': name,
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'parameters': layer_params
                })
        
        print(f"\n📊 Arquitetura da Rede MLP Profunda:")
        print(f"Total de camadas lineares: {layer_count}")
        print(f"Total de parâmetros: {param_count:,}")
        print("-" * 60)
        
        for layer_info in info:
            print(f"Camada {layer_info['layer']:2d}: {layer_info['input_size']:4d} -> "
                  f"{layer_info['output_size']:4d} ({layer_info['parameters']:,} params)")
        
        return info
```

## 📂 Preparação dos Dados para MLP

```python
def create_imbalanced_cifar10(dataset, imbalance_classes, imbalance_ratio=0.1):
    """
    Cria uma versão desbalanceada do CIFAR-10
    """
    indices = []
    class_counts = defaultdict(int)
    
    for i in range(len(dataset)):
        label = dataset.targets[i]
        
        if label in imbalance_classes:
            # Manter apenas uma pequena fração das classes desbalanceadas
            if np.random.rand() < imbalance_ratio:
                indices.append(i)
                class_counts[label] += 1
        else:
            # Manter todas as outras classes
            indices.append(i)
            class_counts[label] += 1
    
    print("Distribuição das classes após desbalanceamento:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Classe {class_id}: {class_counts[class_id]} amostras")
    
    return Subset(dataset, indices), class_counts

def get_data_loaders(config):
    """
    Prepara os DataLoaders com transformações adequadas para MLP
    """
    # Transformações mais simples para MLP (sem data augmentation espacial complexa)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Adicionar ruído como forma de regularização para MLP
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Carregar datasets
    full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    # Criar versão desbalanceada
    imbalanced_train, class_counts = create_imbalanced_cifar10(
        full_train, config['imbalance_classes'], config['imbalance_ratio']
    )
    
    # Calcular pesos para WeightedRandomSampler
    subset_labels = np.array([full_train.targets[i] for i in imbalanced_train.indices])
    class_weights = 1. / np.array([class_counts.get(i, 1) for i in range(10)])
    sample_weights = torch.from_numpy(class_weights[subset_labels]).double()
    
    # Criar samplers
    weighted_sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    
    # DataLoaders
    train_loader_balanced = DataLoader(
        imbalanced_train, 
        batch_size=config['batch_size'], 
        sampler=weighted_sampler
    )
    
    train_loader_unbalanced = DataLoader(
        imbalanced_train, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    return train_loader_balanced, train_loader_unbalanced, test_loader, class_counts
```

## 🎛️ Sistema de Treinamento Avançado

```python
class AdvancedTrainingSystem:
    """
    Sistema completo de treinamento com todas as técnicas integradas para MLPs profundos
    """
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Otimizador com weight decay
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Função de perda
        self.criterion = nn.CrossEntropyLoss()
        
        # Métricas de acompanhamento
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        
        # Monitoramento de gradientes
        self.gradient_norms = []
        
    def monitor_gradients(self):
        """Monitora normas dos gradientes para detectar problemas"""
        total_norm = 0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        return total_norm
    
    def train_epoch(self, dataloader):
        """Treina por uma época com gradient accumulation e monitoramento"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms_epoch = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Normalizar pela acumulação
            loss = loss / self.config['accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                # Monitorar gradientes antes do clipping
                grad_norm = self.monitor_gradients()
                gradient_norms_epoch.append(grad_norm)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_norm']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Métricas
            total_loss += loss.item() * self.config['accumulation_steps']
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Passo final se necessário
        if (batch_idx + 1) % self.config['accumulation_steps'] != 0:
            grad_norm = self.monitor_gradients()
            gradient_norms_epoch.append(grad_norm)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        avg_grad_norm = np.mean(gradient_norms_epoch) if gradient_norms_epoch else 0
        
        return avg_loss, accuracy, avg_grad_norm
    
    def validate(self, dataloader):
        """Validação sem gradient computation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Loop principal de treinamento com early stopping e monitoramento avançado"""
        print("Iniciando treinamento do MLP profundo...")
        start_time = time.time()
        
        # Mostrar informações da arquitetura
        self.model.get_layer_info()
        
        for epoch in range(self.config['num_epochs']):
            # Treinamento
            train_loss, train_acc, grad_norm = self.train_epoch(train_loader)
            
            # Validação
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Salvar métricas
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Log detalhado
            print(f"Época {epoch+1:3d}/{self.config['num_epochs']:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
                  f"LR: {current_lr:.2e} | Grad Norm: {grad_norm:.3f}")
            
            # Early stopping
            if val_loss < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                # Salvar checkpoint
                self.save_checkpoint(epoch, val_loss, 'best_mlp_model.pth')
                
            else:
                self.epochs_no_improve += 1
                
                if self.epochs_no_improve >= self.config['patience']:
                    print(f"\nEarly stopping acionado na época {epoch+1}!")
                    print(f"Melhor perda de validação: {self.best_val_loss:.4f}")
                    break
        
        # Carregar melhor modelo
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        print(f"\nTreinamento concluído em {training_time:.2f} segundos")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def save_checkpoint(self, epoch, loss, filename):
        """Salva checkpoint completo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Carrega checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    
    def evaluate_detailed(self, test_loader, class_names=None):
        """Avaliação detalhada com métricas por classe"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        # Relatório de classificação
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(10)]
        
        print("\n" + "="*50)
        print("RELATÓRIO DETALHADO DE AVALIAÇÃO - MLP PROFUNDO")
        print("="*50)
        print(classification_report(all_targets, all_preds, 
                                  target_names=class_names, digits=4))
        
        # Matriz de confusão
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusão - MLP Profundo')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        plt.show()
        
        return all_preds, all_targets, all_outputs
    
    def plot_training_history(self):
        """Plota histórico de treinamento com métricas específicas para MLP profundo"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Perda
        axes[0, 0].plot(self.train_losses, label='Treinamento', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validação', color='red')
        axes[0, 0].set_title('Perda ao Longo das Épocas')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Perda')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Acurácia
        axes[0, 1].plot(self.train_accuracies, label='Treinamento', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='Validação', color='red')
        axes[0, 1].set_title('Acurácia ao Longo das Épocas')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Acurácia (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[0, 2].plot(self.learning_rates, color='green')
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Época')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # Análise de overfitting
        gap_loss = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1, 0].plot(gap_loss, color='purple')
        axes[1, 0].set_title('Gap de Perda (Validação - Treinamento)')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Diferença de Perda')
        axes[1, 0].grid(True)
        
        # Normas dos gradientes
        if self.gradient_norms:
            axes[1, 1].plot(self.gradient_norms, color='orange')
            axes[1, 1].set_title('Norma dos Gradientes')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Norma L2')
            axes[1, 1].grid(True)
        
        # Análise de convergência
        axes[1, 2].plot(np.diff(self.val_losses), color='brown')
        axes[1, 2].set_title('Taxa de Mudança da Perda de Validação')
        axes[1, 2].set_xlabel('Época')
        axes[1, 2].set_ylabel('Δ Perda de Validação')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
```

## 🚀 Execução do Experimento Completo

```python
def run_mlp_experiment():
    """Executa o experimento completo com MLPs profundos"""
    
    # Preparar dados
    print("Preparando dados para MLPs...")
    train_balanced, train_unbalanced, test_loader, class_counts = get_data_loaders(CONFIG)
    
    # Nomes das classes CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Experimento 1: MLP sem balanceamento
    print("\n" + "="*60)
    print("EXPERIMENTO 1: MLP PROFUNDO SEM BALANCEAMENTO")
    print("="*60)
    
    model1 = DeepMLP(
        input_size=CONFIG['input_size'],
        hidden_sizes=CONFIG['hidden_sizes'],
        num_classes=CONFIG['num_classes'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    system1 = AdvancedTrainingSystem(model1, CONFIG, device)
    history1 = system1.train(train_unbalanced, test_loader)
    system1.plot_training_history()
    
    print("\nAvaliação final - MLP SEM balanceamento:")
    preds1, targets1, outputs1 = system1.evaluate_detailed(test_loader, class_names)
    
    # Experimento 2: MLP com balanceamento
    print("\n" + "="*60)
    print("EXPERIMENTO 2: MLP PROFUNDO COM BALANCEAMENTO")
    print("="*60)
    
    model2 = DeepMLP(
        input_size=CONFIG['input_size'],
        hidden_sizes=CONFIG['hidden_sizes'],
        num_classes=CONFIG['num_classes'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    system2 = AdvancedTrainingSystem(model2, CONFIG, device)
    history2 = system2.train(train_balanced, test_loader)
    system2.plot_training_history()
    
    print("\nAvaliação final - MLP COM balanceamento:")
    preds2, targets2, outputs2 = system2.evaluate_detailed(test_loader, class_names)
    
    # Experimento 3: MLP mais raso para comparação
    print("\n" + "="*60)
    print("EXPERIMENTO 3: MLP MAIS RASO PARA COMPARAÇÃO")
    print("="*60)
    
    shallow_config = CONFIG.copy()
    shallow_config['hidden_sizes'] = [512, 256]  # Apenas 2 camadas escondidas
    
    model3 = DeepMLP(
        input_size=CONFIG['input_size'],
        hidden_sizes=shallow_config['hidden_sizes'],
        num_classes=CONFIG['num_classes'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    system3 = AdvancedTrainingSystem(model3, shallow_config, device)
    history3 = system3.train(train_balanced, test_loader)
    system3.plot_training_history()
    
    print("\nAvaliação final - MLP RASO:")
    preds3, targets3, outputs3 = system3.evaluate_detailed(test_loader, class_names)
    
    # Comparação final
    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL DOS EXPERIMENTOS")
    print("="*60)
    
    compare_mlp_experiments(system1, system2, system3, class_names)
    
    return system1, system2, system3

def compare_mlp_experiments(system1, system2, system3, class_names):
    """Comparação detalhada entre os diferentes experimentos com MLP"""
    
    # Acurácia por classe desbalanceada
    imbalanced_classes = CONFIG['imbalance_classes']
    
    print(f"\nDesempenho nas classes desbalanceadas {imbalanced_classes}:")
    print(f"{'Classe':<12} {'MLP S/ Bal.':<12} {'MLP C/ Bal.':<12} {'MLP Raso':<12}")
    print("-" * 60)
    
    from sklearn.metrics import f1_score
    
    for class_id in imbalanced_classes:
        f1_without = f1_score([1 if t == class_id else 0 for t in system1.evaluate_detailed.__wrapped__(system1, None)[1]], 
                             [1 if p == class_id else 0 for p in system1.evaluate_detailed.__wrapped__(system1, None)[0]], 
                             average='binary')
        # Simplificação para demonstração - em implementação real, usar métricas calculadas anteriormente
        print(f"{class_names[class_id]:<12} {'X.XXXX':<12} {'X.XXXX':<12} {'X.XXXX':<12}")
    
    # Comparação de arquiteturas
    print(f"\n📊 Comparação de Complexidade:")
    print(f"MLP Profundo (6 camadas): {sum(p.numel() for p in system1.model.parameters()):,} parâmetros")
    print(f"MLP Profundo (6 camadas): {sum(p.numel() for p in system2.model.parameters()):,} parâmetros")  
    print(f"MLP Raso (2 camadas):     {sum(p.numel() for p in system3.model.parameters()):,} parâmetros")
    
    # Gráfico comparativo
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Perda de validação
    axes[0, 0].plot(system1.val_losses, label='MLP S/ Balanceamento', alpha=0.7)
    axes[0, 0].plot(system2.val_losses, label='MLP C/ Balanceamento', alpha=0.7)
    axes[0, 0].plot(system3.val_losses, label='MLP Raso', alpha=0.7)
    axes[0, 0].set_title('Comparação: Perda de Validação')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Perda')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Acurácia de validação
    axes[0, 1].plot(system1.val_accuracies, label='MLP S/ Balanceamento', alpha=0.7)
    axes[0, 1].plot(system2.val_accuracies, label='MLP C/ Balanceamento', alpha=0.7)
    axes[0, 1].plot(system3.val_accuracies, label='MLP Raso', alpha=0.7)
    axes[0, 1].set_title('Comparação: Acurácia de Validação')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rates
    axes[1, 0].plot(system1.learning_rates, label='MLP S/ Balanceamento', alpha=0.7)
    axes[1, 0].plot(system2.learning_rates, label='MLP C/ Balanceamento', alpha=0.7)
    axes[1, 0].plot(system3.learning_rates, label='MLP Raso', alpha=0.7)
    axes[1, 0].set_title('Comparação: Learning Rate')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Overfitting comparison
    gap1 = np.array(system1.val_losses) - np.array(system1.train_losses)
    gap2 = np.array(system2.val_losses) - np.array(system2.train_losses)
    gap3 = np.array(system3.val_losses) - np.array(system3.train_losses)
    
    axes[1, 1].plot(gap1, label='MLP S/ Balanceamento', alpha=0.7)
    axes[1, 1].plot(gap2, label='MLP C/ Balanceamento', alpha=0.7)
    axes[1, 1].plot(gap3, label='MLP Raso', alpha=0.7)
    axes[1, 1].set_title('Comparação: Gap de Overfitting')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

system1, system2, system3 = run_mlp_experiment()
```

## 🔍 Análise Específica para MLPs Profundos

```python
def analyze_mlp_depth_effects():
    """Análise específica dos efeitos da profundidade em MLPs"""
    
    print("\n" + "="*60)
    print("ANÁLISE DOS EFEITOS DA PROFUNDIDADE EM MLPs")
    print("="*60)
    
    depths_to_test = [
        [256],                    # 1 camada escondida
        [512, 256],              # 2 camadas escondidas  
        [512, 256, 128],         # 3 camadas escondidas
        [1024, 512, 256, 128],   # 4 camadas escondidas
        [2048, 1024, 512, 256, 128, 64]  # 6 camadas escondidas (original)
    ]
    
    results = {}
    
    for i, hidden_sizes in enumerate(depths_to_test):
        print(f"\nTestando MLP com {len(hidden_sizes)} camada(s) escondida(s)...")
        
        # Configuração para este teste
        test_config = CONFIG.copy()
        test_config['hidden_sizes'] = hidden_sizes
        test_config['num_epochs'] = 20  # Menos épocas para teste rápido
        
        # Criar modelo
        model = DeepMLP(
            input_size=CONFIG['input_size'],
            hidden_sizes=hidden_sizes,
            num_classes=CONFIG['num_classes'],
            dropout_rate=CONFIG['dropout_rate']
        )
        
        # Preparar dados (usar versão balanceada)
        train_balanced, _, test_loader, _ = get_data_loaders(CONFIG)
        
        # Treinar
        system = AdvancedTrainingSystem(model, test_config, device)
        system.train(train_balanced, test_loader)
        
        # Avaliar
        final_acc = max(system.val_accuracies)
        final_loss = min(system.val_losses)
        param_count = sum(p.numel() for p in model.parameters())
        
        results[len(hidden_sizes)] = {
            'accuracy': final_acc,
            'loss': final_loss,
            'parameters': param_count,
            'architecture': hidden_sizes
        }
        
        print(f"  Melhor acurácia: {final_acc:.2f}%")
        print(f"  Parâmetros: {param_count:,}")
    
    # Plotar resultados
    depths = list(results.keys())
    accuracies = [results[d]['accuracy'] for d in depths]
    losses = [results[d]['loss'] for d in depths]
    param_counts = [results[d]['parameters'] for d in depths]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Acurácia vs Profundidade
    axes[0].plot(depths, accuracies, 'bo-')
    axes[0].set_title('Acurácia vs Profundidade')
    axes[0].set_xlabel('Número de Camadas Escondidas')
    axes[0].set_ylabel('Melhor Acurácia (%)')
    axes[0].grid(True)
    
    # Perda vs Profundidade
    axes[1].plot(depths, losses, 'ro-')
    axes[1].set_title('Perda vs Profundidade') 
    axes[1].set_xlabel('Número de Camadas Escondidas')
    axes[1].set_ylabel('Melhor Perda')
    axes[1].grid(True)
    
    # Parâmetros vs Profundidade
    axes[2].plot(depths, param_counts, 'go-')
    axes[2].set_title('Parâmetros vs Profundidade')
    axes[2].set_xlabel('Número de Camadas Escondidas')
    axes[2].set_ylabel('Número de Parâmetros')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Executar análise de profundidade
depth_results = analyze_mlp_depth_effects()
```

## 🎯 Conclusões e Próximos Passos

Este exercício integrado com MLPs profundos demonstra como combinar efetivamente todas as técnicas do Módulo 3:

### ✅ Técnicas Implementadas:
1. **Arquitetura Profunda**: MLP com 6+ camadas escondidas
2. **Preparação de Dados**: Normalização e transformações para MLPs
3. **Balanceamento**: WeightedRandomSampler para classes desbalanceadas
4. **Regularização**: Dropout, Weight Decay, Batch Normalization
5. **Otimização**: Gradient Clipping, Gradient Accumulation, Learning Rate Scheduling
6. **Monitoramento**: Métricas detalhadas, normas de gradientes
7. **Robustez**: Early Stopping, Checkpointing
8. **Avaliação**: Métricas por classe, análise comparativa

### 🔄 Características Específicas para MLPs:

1. **Dimensionalidade**: Trabalha com imagens "achatadas" (3072 features)
2. **Profundidade**: Até 6 camadas escondidas para demonstrar redes profundas
3. **Inicialização**: Xavier/Glorot para melhor convergência
4. **Monitoramento**: Acompanhamento específico de normas de gradientes

### 📝 Exercícios Adicionais:

1. **Experimente diferentes profundidades**: 2, 4, 8, 10 camadas
2. **Teste diferentes funções de ativação**: ReLU, LeakyReLU, ELU, Swish
3. **Implemente diferentes inicializações**: He, Xavier normal vs uniform
4. **Adicione outras técnicas de regularização**: Layer normalization, spectral normalization
5. **Compare com arquiteturas alternativas**: ResNet-style skip connections para MLPs

### 🚨 Desafios Comuns com MLPs Profundos:

1. **Vanishing Gradients**: Monitorado através das normas de gradientes
2. **Overfitting**: Controlado com dropout e weight decay
3. **Lenta Convergência**: Gerenciada com learning rate scheduling
4. **Instabilidade**: Tratada com gradient clipping e batch normalization

Este exercício serve como base sólida para entender como treinar MLPs profundos efetivamente antes de avançar para arquiteturas mais complexas como CNNs.