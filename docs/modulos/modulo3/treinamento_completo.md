---
sidebar_position: 9
title: "Exercício com Treinamento completo"
description: "Este exercício integra todas as técnicas abordadas no Módulo 3, criando um sistema completo de treinamento de redes neurais com PyTorch."
tags: [dropout, weight decay, batch normalization]
---

# Exercício Prático Integrado: Sistema Completo de Treinamento

Este exercício integra todas as técnicas abordadas no Módulo 3, criando um sistema completo de treinamento de redes neurais com PyTorch.

## 🎯 Objetivo

Implementar um sistema de treinamento robusto que inclui:
- Carregamento e preparação de dados
- Tratamento de desbalanceamento
- Múltiplas técnicas de regularização
- Monitoramento e diagnóstico
- Checkpointing e inferência otimizada

## 📊 Dataset: CIFAR-10 Desbalanceado

Vamos criar uma versão artificialmente desbalanceada do CIFAR-10 para simular um cenário real desafiador.

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
    'weight_decay': 1e-4
}
```

## 🏗️ Arquitetura do Modelo

```python
class AdvancedCNN(nn.Module):
    """
    CNN avançada com múltiplas técnicas de regularização
    """
    def __init__(self, num_classes=10, dropout_rate=0.3, use_batch_norm=True):
        super(AdvancedCNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        
        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        
        # Segunda camada convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        
        # Terceira camada convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512) if use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Pooling e ativação
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Bloco 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Bloco 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Camadas FC
        x = self.dropout1(self.relu(self.bn4(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn5(self.fc2(x))))
        x = self.fc3(x)
        
        return x
```

## 📂 Preparação dos Dados

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
    Prepara os DataLoaders com transformações e balanceamento
    """
    # Transformações
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

## 🎛️ Sistema de Treinamento

```python
class TrainingSystem:
    """
    Sistema completo de treinamento com todas as técnicas integradas
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
        
        # Função de perda
        self.criterion = nn.CrossEntropyLoss()
        
        # Métricas de acompanhamento
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        
    def train_epoch(self, dataloader):
        """Treina por uma época com gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
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
        """Loop principal de treinamento com early stopping"""
        print("Iniciando treinamento...")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Treinamento
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validação
            val_loss, val_acc = self.validate(val_loader)
            
            # Salvar métricas
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log
            print(f"Época {epoch+1:3d}/{self.config['num_epochs']:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%")
            
            # Early stopping
            if val_loss < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                # Salvar checkpoint
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')
                
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
            'loss': loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Carrega checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    
    def evaluate_detailed(self, test_loader, class_names=None):
        """Avaliação detalhada com métricas por classe"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Relatório de classificação
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(10)]
        
        print("\n" + "="*50)
        print("RELATÓRIO DETALHADO DE AVALIAÇÃO")
        print("="*50)
        print(classification_report(all_targets, all_preds, 
                                  target_names=class_names, digits=4))
        
        # Matriz de confusão
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.tight_layout()
        plt.show()
        
        return all_preds, all_targets
    
    def plot_training_history(self):
        """Plota histórico de treinamento"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Análise de overfitting
        gap_loss = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1, 0].plot(gap_loss, color='purple')
        axes[1, 0].set_title('Gap de Perda (Validação - Treinamento)')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Diferença de Perda')
        axes[1, 0].grid(True)
        
        # Análise de convergência
        axes[1, 1].plot(np.diff(self.val_losses), color='orange')
        axes[1, 1].set_title('Taxa de Mudança da Perda de Validação')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Δ Perda de Validação')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
```

## 🚀 Execução do Experimento

```python
def run_complete_experiment():
    """Executa o experimento completo"""
    
    # Preparar dados
    print("Preparando dados...")
    train_balanced, train_unbalanced, test_loader, class_counts = get_data_loaders(CONFIG)
    
    # Nomes das classes CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Experimento 1: Sem balanceamento
    print("\n" + "="*60)
    print("EXPERIMENTO 1: TREINAMENTO SEM BALANCEAMENTO")
    print("="*60)
    
    model1 = AdvancedCNN(dropout_rate=CONFIG['dropout_rate'])
    system1 = TrainingSystem(model1, CONFIG, device)
    
    history1 = system1.train(train_unbalanced, test_loader)
    system1.plot_training_history()
    
    print("\nAvaliação final - Modelo SEM balanceamento:")
    preds1, targets1 = system1.evaluate_detailed(test_loader, class_names)
    
    # Experimento 2: Com balanceamento
    print("\n" + "="*60)
    print("EXPERIMENTO 2: TREINAMENTO COM BALANCEAMENTO")
    print("="*60)
    
    model2 = AdvancedCNN(dropout_rate=CONFIG['dropout_rate'])
    system2 = TrainingSystem(model2, CONFIG, device)
    
    history2 = system2.train(train_balanced, test_loader)
    system2.plot_training_history()
    
    print("\nAvaliação final - Modelo COM balanceamento:")
    preds2, targets2 = system2.evaluate_detailed(test_loader, class_names)
    
    # Comparação final
    print("\n" + "="*60)
    print("COMPARAÇÃO FINAL DOS EXPERIMENTOS")
    print("="*60)
    
    # Acurácia por classe desbalanceada
    imbalanced_classes = CONFIG['imbalance_classes']
    
    print(f"\nDesempenho nas classes desbalanceadas {imbalanced_classes}:")
    print(f"{'Classe':<10} {'Sem Balanc.':<12} {'Com Balanc.':<12} {'Melhoria':<10}")
    print("-" * 50)
    
    for class_id in imbalanced_classes:
        # Calcular F1-score por classe
        from sklearn.metrics import f1_score
        f1_without = f1_score(targets1, preds1, labels=[class_id], average=None)[0]
        f1_with = f1_score(targets2, preds2, labels=[class_id], average=None)[0]
        improvement = ((f1_with - f1_without) / f1_without * 100) if f1_without > 0 else 0
        
        print(f"{class_names[class_id]:<10} {f1_without:<12.4f} {f1_with:<12.4f} {improvement:<10.1f}%")
    
    return system1, system2

# Executar experimento
if __name__ == "__main__":
    system1, system2 = run_complete_experiment()
```

## 📊 Análise de Resultados

```python
def analyze_results(system1, system2):
    """Análise comparativa detalhada dos resultados"""
    
    print("\n" + "="*60)
    print("ANÁLISE DETALHADA DOS RESULTADOS")
    print("="*60)
    
    # Comparar curvas de aprendizado
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Perda de validação
    axes[0, 0].plot(system1.val_losses, label='Sem Balanceamento', color='red', alpha=0.7)
    axes[0, 0].plot(system2.val_losses, label='Com Balanceamento', color='blue', alpha=0.7)
    axes[0, 0].set_title('Comparação: Perda de Validação')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Perda')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Acurácia de validação
    axes[0, 1].plot(system1.val_accuracies, label='Sem Balanceamento', color='red', alpha=0.7)
    axes[0, 1].plot(system2.val_accuracies, label='Com Balanceamento', color='blue', alpha=0.7)
    axes[0, 1].set_title('Comparação: Acurácia de Validação')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Acurácia (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Estabilidade do treinamento
    stability1 = np.std(system1.val_losses[-10:])  # Desvio padrão das últimas 10 épocas
    stability2 = np.std(system2.val_losses[-10:])
    
    axes[1, 0].bar(['Sem Balanceamento', 'Com Balanceamento'], 
                   [stability1, stability2], 
                   color=['red', 'blue'], alpha=0.7)
    axes[1, 0].set_title('Estabilidade do Treinamento\n(Desvio Padrão da Perda)')
    axes[1, 0].set_ylabel('Desvio Padrão')
    
    # Convergência
    final_loss1 = min(system1.val_losses)
    final_loss2 = min(system2.val_losses)
    
    axes[1, 1].bar(['Sem Balanceamento', 'Com Balanceamento'], 
                   [final_loss1, final_loss2], 
                   color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_title('Melhor Perda de Validação Alcançada')
    axes[1, 1].set_ylabel('Perda')
    
    plt.tight_layout()
    plt.show()
    
    # Relatório de insights
    print("\n📈 INSIGHTS PRINCIPAIS:")
    print("-" * 40)
    
    if final_loss2 < final_loss1:
        print("✅ O balanceamento de classes resultou em melhor convergência")
    else:
        print("❌ O balanceamento não melhorou a convergência geral")
    
    if stability2 < stability1:
        print("✅ O treinamento com balanceamento foi mais estável")
    else:
        print("❌ O balanceamento não melhorou a estabilidade")
    
    # Análise por época
    epochs_to_best1 = np.argmin(system1.val_losses) + 1
    epochs_to_best2 = np.argmin(system2.val_losses) + 1
    
    print(f"\n⏱️  VELOCIDADE DE CONVERGÊNCIA:")
    print(f"   Sem balanceamento: {epochs_to_best1} épocas para melhor resultado")
    print(f"   Com balanceamento: {epochs_to_best2} épocas para melhor resultado")
    
    if epochs_to_best2 < epochs_to_best1:
        print("✅ O balanceamento acelerou a convergência")
    else:
        print("❌ O balanceamento não acelerou a convergência")

# Executar análise
analyze_results(system1, system2)
```

## 🎯 Conclusões e Próximos Passos

Este exercício integrado demonstra como combinar efetivamente todas as técnicas do Módulo 3:

### ✅ Técnicas Implementadas:
1. **Preparação de Dados**: CustomDataset e DataLoader
2. **Balanceamento**: WeightedRandomSampler
3. **Regularização**: Dropout, Weight Decay, Batch Normalization
4. **Otimização**: Gradient Clipping, Gradient Accumulation
5. **Monitoramento**: Métricas detalhadas, visualizações
6. **Robustez**: Early Stopping, Checkpointing
7. **Avaliação**: Métricas por classe, análise comparativa

### 🔄 Extensões Possíveis:

1. **Diferentes Arquiteturas**: Teste com ResNet, DenseNet
2. **Hiperparâmetros**: Grid search automatizado
3. **Augmentação**: Técnicas mais avançadas
4. **Ensemble**: Combinação de modelos
5. **Transfer Learning**: Modelos pré-treinados

### 📝 Exercícios Adicionais:

1. Modifique o dataset para simular outros tipos de desbalanceamento
2. Implemente outras métricas (AUC, F1 macro/micro)
3. Adicione visualizações de ativações
4. Teste diferentes schedulers de learning rate
5. Implemente cross-validation

Este exercício serve como base sólida para projetos reais de Deep Learning, integrando as melhores práticas apresentadas no módulo.