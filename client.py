from typing import Dict, Any
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from shap_utils import batch_importance_weights


class Client:
    def __init__(self, cid: int, model: nn.Module, train_loader, device: torch.device, cluster_id: int):
        self.id = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.cluster_id = cluster_id
        self.local_metrics = []  # list of dicts: {client_id, epoch, loss, acc}

    def train_local(self, epochs: int = CFG.local_epochs, lr: float = CFG.lr_client, round_num: int = 0, logger=None):
        start_time = time.time()
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
        for e in range(1, epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0.0
            total = 0
            correct = 0
            step = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = torch.nn.functional.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total += batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                
                # Log step-level metrics if logger is provided
                if logger:
                    step_acc = (preds == y).float().mean().item()
                    logger.log_client_step_metrics(
                        client_id=self.id,
                        round_num=round_num,
                        epoch=e,
                        step=step,
                        loss=loss.item(),
                        accuracy=step_acc
                    )
                step += 1

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_loss = total_loss / max(1, total)
            epoch_acc = correct / max(1, total)
            
            # Store local metrics
            epoch_metrics = {
                "client_id": self.id,
                "cluster_id": self.cluster_id,
                "round": round_num,
                "epoch": e,
                "loss": epoch_loss,
                "acc": epoch_acc,
                "epoch_time": epoch_time,
            }
            self.local_metrics.append(epoch_metrics)
            
            # Log epoch-level metrics to Excel immediately
            if logger:
                logger.log_client_epoch_metrics(
                    client_id=self.id,
                    cluster_id=self.cluster_id,
                    round_num=round_num,
                    epoch=e,
                    loss=epoch_loss,
                    accuracy=epoch_acc,
                    epoch_time=epoch_time
                )
            
            print(f"[Client {self.id}] Local epoch {e}/{epochs} - loss: {epoch_loss:.4f}, acc: {epoch_acc*100:.2f}%, time: {epoch_time:.2f}s")
        
        total_training_time = time.time() - start_time
        
        # Log total client training time
        if logger:
            logger.log_client_training_time(
                client_id=self.id,
                cluster_id=self.cluster_id,
                round_num=round_num,
                total_time=total_training_time
            )
        
        return self.local_metrics, total_training_time

    @torch.no_grad()
    def produce_signals(self, public_loader, current_round: int = 0) -> Dict[str, Any]:
        """Evaluate teacher on public data and produce logits, features, and importance weights."""
        self.model.eval()
        teacher_logits = []
        teacher_feats = []
        shap_weights = []

        try:
            for batch_idx, (x, y) in enumerate(public_loader):
                try:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Forward pass
                    out = self.model(x)
                    if isinstance(out, (tuple, list)):
                        logits, feats = out
                        teacher_feats.append(feats.detach().cpu())
                    else:
                        logits = out
                        teacher_feats.append(None)
                    teacher_logits.append(logits.detach().cpu())
                    
                    # Importance weights computation with error handling
                    try:
                        # Temporarily enable gradients for SHAP computation
                        with torch.enable_grad():
                            x_grad = x.detach().requires_grad_(True)
                            imp = batch_importance_weights(self.model, x_grad, y)
                        shap_weights.append(imp.detach().cpu())
                    except Exception as shap_error:
                        print(f"[WARNING] Client {self.id} SHAP computation failed for batch {batch_idx}: {str(shap_error)}")
                        # Fallback to uniform weights
                        batch_size = x.size(0)
                        fallback_weights = torch.ones(batch_size) / batch_size
                        shap_weights.append(fallback_weights)
                    
                    # Clear GPU memory after each batch
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as batch_error:
                    print(f"[ERROR] Client {self.id} failed processing batch {batch_idx}: {str(batch_error)}")
                    import traceback
                    traceback.print_exc()
                    # Skip this batch and continue
                    continue

            # Calculate number of samples in client's training data
            num_samples = len(self.train_loader.dataset)
            
            # Get the latest training accuracy and loss from local metrics
            latest_accuracy = 0.1  # Default fallback accuracy
            latest_loss = 1.0  # Default fallback loss
            if self.local_metrics:
                # Get the accuracy and loss from the last epoch of the most recent round
                latest_metrics = [m for m in self.local_metrics if m["round"] == current_round]
                if latest_metrics:
                    # Use the accuracy and loss from the last epoch
                    latest_accuracy = latest_metrics[-1]["acc"]
                    latest_loss = latest_metrics[-1]["loss"]

            return {
                "client_id": self.id,
                "cluster_id": self.cluster_id,
                "logits": teacher_logits,  # list of tensors per batch
                "features": teacher_feats,  # list of tensors per batch
                "importance": shap_weights,  # list of tensors per batch
                "model_state": self.model.state_dict(),  # model parameters for FedAvg
                "num_samples": num_samples,  # number of training samples for weighted averaging
                "accuracy": latest_accuracy,  # client's latest training accuracy for weighted aggregation
                "loss": latest_loss,  # client's latest training loss for display
            }
            
        except Exception as e:
            print(f"[ERROR] Client {self.id} signal generation completely failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return minimal signal to prevent complete failure
            num_samples = len(self.train_loader.dataset)
            return {
                "client_id": self.id,
                "cluster_id": self.cluster_id,
                "logits": [],
                "features": [],
                "importance": [],
                "model_state": self.model.state_dict(),
                "num_samples": num_samples,
                "accuracy": 0.1,  # Default fallback accuracy
                "loss": 1.0,  # Default fallback loss
            }