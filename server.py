from typing import List, Dict, Any
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import CFG
from models import build_adaptive_model, build_adaptive_student
from distillation import (
    confidence_weights,
    aggregate_teacher_logits,
    feature_alignment_loss,
    total_loss,
    kd_loss,
    importance_weighted_kd_loss,
)
from utils import accuracy_from_logits


class ClusterServer:
    """Server that manages cluster-specific global models and multi-teacher knowledge distillation."""
    
    def __init__(self, device: torch.device, num_clusters: int = 4, num_classes: int = 10, 
                 in_channels: int = 1, image_size: int = 28, dataset_name: str = "MNIST"):
        self.device = device
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        
        # Create cluster-specific global models with adaptive input channels
        self.cluster_models = {}
        for cluster_id in range(num_clusters):
            model = build_adaptive_model(cluster_id, dataset_name, num_classes, image_size)
            self.cluster_models[cluster_id] = model.to(device)
        
        # Create student model for multi-teacher KD with adaptive input channels
        self.student = build_adaptive_student(dataset_name, num_classes, image_size).to(device)
        
        print(f"[ClusterServer] Initialized with {num_clusters} cluster models and 1 student model for {dataset_name}")
    
    def train_cluster_models(self, client_signals: List[Dict[str, Any]], public_loader, 
                           epochs: int = CFG.distill_epochs, current_round: int = 0, logger=None):
        """Aggregate client models using FedAvg for each cluster (no additional training)."""
        
        # Group client signals by cluster
        cluster_signals = defaultdict(list)
        for signal in client_signals:
            cluster_id = signal["cluster_id"]
            cluster_signals[cluster_id].append(signal)
        
        cluster_summaries = []
        
        for cluster_id in range(self.num_clusters):
            if cluster_id not in cluster_signals:
                print(f"[ClusterServer] No clients in cluster {cluster_id}, skipping...")
                continue
            
            print(f"[ClusterServer] Aggregating cluster {cluster_id} model with {len(cluster_signals[cluster_id])} clients")
            
            # Apply FedAvg aggregation for this cluster (this is the core FedAvg step)
            cluster_avg_metrics = self._fedavg_aggregate_cluster(cluster_id, cluster_signals[cluster_id], current_round)
            
            # Store cluster average metrics for Excel logging if available
            if cluster_avg_metrics and logger:
                logger.log_cluster_aggregation_metrics(
                    round_num=current_round,
                    cluster_id=cluster_id,
                    avg_loss=cluster_avg_metrics["avg_loss"],
                    avg_accuracy=cluster_avg_metrics["avg_accuracy"],
                    num_clients=cluster_avg_metrics["num_clients"]
                )
            
            # Prepare cluster summary for round logging (evaluate aggregated model)
            model = self.cluster_models[cluster_id]
            model.eval()
            
            # Quick evaluation on a small batch to get metrics for logging
            running_loss = 0.0
            total = 0
            correct = 0
            
            with torch.no_grad():
                for i, (x, y) in enumerate(public_loader):
                    if i >= 5:  # Only evaluate on first 5 batches for efficiency
                        break
                    x, y = x.to(self.device), y.to(self.device)
                    
                    model_out = model(x)
                    if isinstance(model_out, (tuple, list)):
                        cluster_logits, _ = model_out
                    else:
                        cluster_logits = model_out
                    
                    loss = F.cross_entropy(cluster_logits, y)
                    
                    batch_size = y.size(0)
                    running_loss += loss.item() * batch_size
                    total += batch_size
                    preds = cluster_logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
            
            if total > 0:
                cluster_summary = {
                    "cluster_id": cluster_id,
                    "num_clients": len(cluster_signals[cluster_id]),
                    "avg_loss": running_loss / total,
                    "avg_accuracy": correct / total
                }
                cluster_summaries.append(cluster_summary)
                # Removed the aggregated model print statement as requested
        
        # Log round summary for all clusters - REMOVED as requested
        # if logger and cluster_summaries:
        #     logger.log_round_summary_metrics(
        #         round_num=current_round,
        #         cluster_summaries=cluster_summaries
        #     )
        
        # Return cluster summaries for compatibility with add_distill
        return cluster_summaries
        
    def train_cluster_models_on_public_dataset(self, public_loader, dataset_name: str, epochs: int = None, logger=None):
        """Train each cluster model on public dataset (FashionMNIST or CIFAR-10) for post-hoc refinement."""
        
        # Get dataset-specific configuration
        dataset_config = CFG.dataset_configs.get(dataset_name, CFG.dataset_configs["FashionMNIST"])
        if epochs is None:
            epochs = dataset_config["distill_epochs"]
        
        print(f"[ClusterServer] Post-hoc training of cluster models on {dataset_name} for {epochs} epochs")
        print(f"[ClusterServer] Using dataset config: lr={dataset_config['lr_server']}, batch_size={dataset_config['batch_size']}")
        
        # Start timing cluster training
        cluster_training_start_time = time.time()
        
        cluster_stats = []
        
        for cluster_id in range(self.num_clusters):
            print(f"[ClusterServer] Training cluster {cluster_id} model on {dataset_name}...")
            
            model = self.cluster_models[cluster_id]
            model.train()
            
            optimizer = optim.Adam(model.parameters(), lr=dataset_config["lr_server"], weight_decay=CFG.weight_decay)
            
            epoch_stats = []
            for e in range(1, epochs + 1):
                running_loss = 0.0
                total = 0
                correct = 0
                
                for x, y in public_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    model_out = model(x)
                    if isinstance(model_out, (tuple, list)):
                        logits, _ = model_out
                    else:
                        logits = model_out
                    
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    
                    batch_size = y.size(0)
                    running_loss += loss.item() * batch_size
                    total += batch_size
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                
                epoch_loss = running_loss / total
                epoch_acc = correct / total
                
                epoch_stat = {
                    "cluster_id": cluster_id,
                    "epoch": e,
                    "loss": epoch_loss,
                    "accuracy": epoch_acc
                }
                epoch_stats.append(epoch_stat)
                
                print(f"[ClusterServer] Cluster {cluster_id} epoch {e}/{epochs} on {dataset_name} - loss: {epoch_loss:.4f}, acc: {epoch_acc*100:.2f}%")
                
                # Log post-hoc cluster training metrics to Excel
                if logger:
                    logger.log_posthoc_distillation_metrics(
                        cluster_id=cluster_id,
                        epoch=e,
                        loss=epoch_loss,
                        accuracy=epoch_acc,
                        phase=f"cluster_training_{dataset_name.lower()}"
                    )
            
            cluster_stats.append({
                "cluster_id": cluster_id,
                "epochs": epoch_stats
            })
        
        # End timing cluster training
        cluster_training_end_time = time.time()
        cluster_training_duration = cluster_training_end_time - cluster_training_start_time
        print(f"[ClusterServer] Cluster models training on {dataset_name} completed in {cluster_training_duration:.2f} seconds")
        
        return cluster_stats
    
    def train_cluster_models_on_fashionmnist(self, fashion_mnist_loader, epochs: int = 5, logger=None):
        """Legacy function for backward compatibility. Use train_cluster_models_on_public_dataset instead."""
        return self.train_cluster_models_on_public_dataset(fashion_mnist_loader, "FashionMNIST", epochs, logger)
    
    def train_student_with_teachers(self, public_loader, dataset_name: str = "FashionMNIST", epochs: int = None, current_round: int = 0, logger=None, client_signals: List[Dict[str, Any]] = None):
        """Train student model using cluster models as teachers (multi-teacher KD)."""
        
        # Get dataset-specific configuration
        dataset_config = CFG.dataset_configs.get(dataset_name, CFG.dataset_configs["FashionMNIST"])
        if epochs is None:
            epochs = dataset_config["distill_epochs"]
        
        print(f"[ClusterServer] Training student with {self.num_clusters} cluster teachers on {dataset_name}")
        print(f"[ClusterServer] Using dataset config: lr={dataset_config['lr_server']}, temperature={dataset_config['temperature']}")
        
        # Extract and prepare importance weights from client signals if available
        importance_weights_dict = {}
        if client_signals:
            for signal in client_signals:
                if "importance" in signal:
                    importance_weights_dict[signal["client_id"]] = signal["importance"]
        
        self.student.train()
        # Set cluster models to eval mode for teacher inference
        for model in self.cluster_models.values():
            model.eval()
        
        optimizer = optim.Adam(self.student.parameters(), lr=dataset_config["lr_server"], weight_decay=CFG.weight_decay)
        
        epoch_stats = []
        for e in range(1, epochs + 1):
            running_loss = 0.0
            total = 0
            correct = 0
            step = 0
            
            for x, y in public_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Get teacher logits from all cluster models
                teacher_logits = []
                teacher_features = []
                
                with torch.no_grad():
                    for cluster_id in range(self.num_clusters):
                        teacher_out = self.cluster_models[cluster_id](x)
                        if isinstance(teacher_out, (tuple, list)):
                            t_logits, t_feats = teacher_out
                            teacher_features.append(t_feats)
                        else:
                            t_logits = teacher_out
                            teacher_features.append(None)
                        teacher_logits.append(t_logits)
                
                # Student forward pass
                student_out = self.student(x)
                if isinstance(student_out, (tuple, list)):
                    student_logits, student_feats = student_out
                else:
                    student_logits = student_out
                    student_feats = None
                
                # Multi-teacher knowledge distillation
                # Simple equal weighting of teachers
                weights = torch.ones(len(teacher_logits), device=self.device) / len(teacher_logits)
                
                # Aggregate teacher logits
                aggregated_teacher_logits = aggregate_teacher_logits(teacher_logits, weights)
                
                # Feature alignment loss (if available)
                if student_feats is not None and any(tf is not None for tf in teacher_features):
                    feat_loss = feature_alignment_loss([student_feats], teacher_features, weights)
                else:
                    feat_loss = torch.tensor(0.0, device=self.device)
                
                # Prepare importance weights for this batch if available
                batch_importance_weights = None
                if importance_weights_dict:
                    # For simplicity, use the first available importance weights
                    # In practice, you might want to aggregate or select based on cluster assignment
                    first_client_id = next(iter(importance_weights_dict.keys()))
                    client_importance = importance_weights_dict[first_client_id]
                    if len(client_importance) >= batch_size:
                        batch_importance_weights = torch.tensor(client_importance[:batch_size], 
                                                              device=self.device, dtype=torch.float32)
                
                # Total loss: KD + CE + feature alignment with importance weighting
                loss = total_loss(student_logits, aggregated_teacher_logits, student_logits, y, 
                                feat_loss, T=dataset_config["temperature"], importance_weights=batch_importance_weights)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_size = y.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size
                preds = student_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                
                # Log step-level metrics if logger is provided
                if logger:
                    step_acc = (preds == y).float().mean().item()
                    logger.log_server_step_metrics(
                        round_num=current_round,
                        epoch=e,
                        step=step,
                        loss=loss.item(),
                        accuracy=step_acc,
                        model_type="student_multiteacher"
                    )
                step += 1
            
            epoch_record = {
                "round": current_round,
                "epoch": e,
                "loss": running_loss / max(1, total),
                "acc": correct / max(1, total),
            }
            print(f"[ClusterServer] Student epoch {e}/{epochs} - loss: {epoch_record['loss']:.4f}, acc: {epoch_record['acc']*100:.2f}%")
            epoch_stats.append(epoch_record)
            
            # Log post-hoc student distillation metrics to Excel
            if logger:
                logger.log_posthoc_distillation_metrics(
                    cluster_id=-1,  # Use -1 to indicate student model (not cluster-specific)
                    epoch=e,
                    loss=epoch_record['loss'],
                    accuracy=epoch_record['acc'],
                    phase="student_distillation"
                )
        
        return epoch_stats
    
    def train_global_student(self, public_loader, epochs: int = 10):
        """Train student model globally on public dataset after all federated rounds complete."""
        
        print(f"[ClusterServer] Global student training on public dataset for {epochs} epochs")
        
        self.student.train()
        # Set cluster models to eval mode for teacher inference
        for model in self.cluster_models.values():
            model.eval()
        
        optimizer = optim.Adam(self.student.parameters(), lr=CFG.lr_server, weight_decay=CFG.weight_decay)
        
        epoch_stats = []
        for e in range(1, epochs + 1):
            running_loss = 0.0
            total = 0
            correct = 0
            
            for x, y in public_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Get teacher logits from all cluster models
                teacher_logits = []
                teacher_features = []
                
                with torch.no_grad():
                    for cluster_id in range(self.num_clusters):
                        teacher_out = self.cluster_models[cluster_id](x)
                        if isinstance(teacher_out, (tuple, list)):
                            t_logits, t_feats = teacher_out
                            teacher_features.append(t_feats)
                        else:
                            t_logits = teacher_out
                            teacher_features.append(None)
                        teacher_logits.append(t_logits)
                
                # Student forward pass
                student_out = self.student(x)
                if isinstance(student_out, (tuple, list)):
                    student_logits, student_feats = student_out
                else:
                    student_logits = student_out
                    student_feats = None
                
                # Multi-teacher knowledge distillation
                # Simple equal weighting of teachers
                weights = torch.ones(len(teacher_logits), device=self.device) / len(teacher_logits)
                
                # Aggregate teacher logits
                aggregated_teacher_logits = aggregate_teacher_logits(teacher_logits, weights)
                
                # Feature alignment loss (if available)
                if student_feats is not None and any(tf is not None for tf in teacher_features):
                    feat_loss = feature_alignment_loss([student_feats], teacher_features, weights)
                else:
                    feat_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss: KD + CE + feature alignment
                loss = total_loss(student_logits, aggregated_teacher_logits, student_logits, y, 
                                feat_loss, T=CFG.temperature)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                preds = student_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            epoch_acc = correct / max(1, total)
            epoch_loss = running_loss / len(public_loader)
            
            print(f"[ClusterServer] Global training epoch {e}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
            
            epoch_stats.append({
                "round": -1,  # Special round number for global training
                "epoch": e,
                "loss": epoch_loss,
                "acc": epoch_acc
            })
        
        print(f"[ClusterServer] Global student training completed")
        return epoch_stats

    @torch.no_grad()
    def evaluate_student(self, test_loader):
        """Evaluate student model on test data."""
        self.student.eval()
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.student(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        return correct / max(1, total)
    
    @torch.no_grad()
    def evaluate_cluster_model(self, cluster_id: int, test_loader):
        """Evaluate specific cluster model on test data."""
        if cluster_id not in self.cluster_models:
            return 0.0
        
        model = self.cluster_models[cluster_id]
        model.eval()
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        return correct / max(1, total)
    
    def _fedavg_aggregate_cluster(self, cluster_id: int, cluster_signals: List[Dict[str, Any]], current_round: int = None):
        """Apply accuracy-weighted FedAvg aggregation to update the cluster's global model."""
        if not cluster_signals:
            return
        
        cluster_model = self.cluster_models[cluster_id]
        
        # Calculate weights based on both accuracy and number of samples
        total_weighted_score = 0.0
        client_weights = []
        
        for signal in cluster_signals:
            # Get client accuracy (default to 0.1 if not available to avoid zero weights)
            client_accuracy = signal.get("accuracy", 0.1)
            num_samples = signal["num_samples"]
            
            # Combine accuracy and sample size for weighting
            # Higher accuracy clients get more weight, scaled by their data size
            weighted_score = client_accuracy * num_samples
            client_weights.append(weighted_score)
            total_weighted_score += weighted_score
        
        # Normalize weights
        if total_weighted_score > 0:
            client_weights = [w / total_weighted_score for w in client_weights]
        else:
            # Fallback to equal weights if all accuracies are zero
            client_weights = [1.0 / len(cluster_signals)] * len(cluster_signals)
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        for i, signal in enumerate(cluster_signals):
            client_state = signal["model_state"]
            weight = client_weights[i]
            
            if i == 0:
                # Initialize with first client's weighted parameters
                for key, param in client_state.items():
                    aggregated_state[key] = param.clone() * weight
            else:
                # Add weighted parameters from subsequent clients
                for key, param in client_state.items():
                    aggregated_state[key] += param * weight
        
        # Update cluster model with aggregated parameters
        cluster_model.load_state_dict(aggregated_state)
        
        # Print detailed aggregation info without weights
        print(f"[ClusterServer] Accuracy-weighted FedAvg aggregation completed for cluster {cluster_id}")
        
        # Calculate and display cluster averages
        total_loss = 0.0
        total_accuracy = 0.0
        for i, signal in enumerate(cluster_signals):
            client_id = signal.get("client_id", i)
            accuracy = signal.get("accuracy", 0.0)
            loss = signal.get("loss", 0.0)  # Get loss from signal if available
            print(f"  Client {client_id}: loss={loss:.4f}, accuracy={accuracy:.4f}")
            total_loss += loss
            total_accuracy += accuracy
        
        # Display cluster averages with round number
        avg_loss = total_loss / len(cluster_signals) if cluster_signals else 0.0
        avg_accuracy = total_accuracy / len(cluster_signals) if cluster_signals else 0.0
        round_info = f"Round {current_round} - " if current_round is not None else ""
        print(f"[ClusterServer] {round_info}Cluster {cluster_id} FedAvg - loss: {avg_loss:.4f}, accuracy: {avg_accuracy:.4f}")
        print(f"[ClusterServer] Total clients aggregated: {len(cluster_signals)}")
        
        # Return cluster averages for Excel storage
        return {
            "cluster_id": cluster_id,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "num_clients": len(cluster_signals)
        }


# Legacy Server class for backward compatibility
class Server:
    def __init__(self, student: nn.Module, device: torch.device):
        self.student = student.to(device)
        self.device = device

    def distill(self, teacher_packages: List[Dict[str, Any]], public_loader, epochs: int = CFG.distill_epochs, current_round: int = 0):
        # Legacy implementation - simplified version
        self.student.train()
        optimizer = optim.Adam(self.student.parameters(), lr=CFG.lr_server, weight_decay=CFG.weight_decay)
        
        epoch_stats = []
        for e in range(1, epochs + 1):
            running_loss = 0.0
            total = 0
            correct = 0
            
            for x, y in public_loader:
                x, y = x.to(self.device), y.to(self.device)
                student_logits, _ = self.student(x)
                loss = F.cross_entropy(student_logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_size = y.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size
                preds = student_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
            
            epoch_record = {
                "round": current_round,
                "epoch": e,
                "loss": running_loss / max(1, total),
                "acc": correct / max(1, total),
            }
            epoch_stats.append(epoch_record)
        
        return epoch_stats

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.student.eval()
        correct = 0
        total = 0
        
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.student(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        return correct / max(1, total)