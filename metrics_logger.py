from typing import List, Dict

import os
import pandas as pd


class MetricsLogger:
    def __init__(self):
        self.local_records: List[Dict] = []
        self.distill_records: List[Dict] = []
        self.round_eval_records: List[Dict] = []
        self.client_step_records: List[Dict] = []  # For detailed client training metrics
        self.server_step_records: List[Dict] = []  # For detailed server training metrics
        self.cluster_eval_records: List[Dict] = []  # For cluster evaluation metrics
        self.global_aggregation_records: List[Dict] = []  # For global aggregation metrics
        self.client_epoch_records: List[Dict] = []  # For client epoch-level metrics
        self.round_summary_records: List[Dict] = []  # For round-level summary metrics
        self.cluster_aggregation_records: List[Dict] = []  # For cluster aggregation average metrics
        self.posthoc_distillation_records: List[Dict] = []  # For post-hoc knowledge distillation metrics
        self.timing_records: List[Dict] = []  # For timing measurements

    def add_local(self, records: List[Dict]):
        self.local_records.extend(records)

    def add_distill(self, records: List[Dict]):
        self.distill_records.extend(records)

    def add_round_eval(self, round_idx: int, test_acc: float):
        self.round_eval_records.append({"round": round_idx, "test_acc": test_acc})

    def log_server_metrics(self, stat: Dict):
        """Log server-side training metrics (e.g., student distillation stats)."""
        self.add_distill([stat])

    def log_evaluation_metrics(self, eval_record: Dict):
        """Log evaluation metrics for models on test datasets."""
        self.add_round_eval(eval_record["round"], eval_record["accuracy"])

    def log_client_step_metrics(self, client_id: int, round_num: int, epoch: int, step: int, loss: float, accuracy: float = None):
        """Log detailed client training metrics for every step."""
        record = {
            "client_id": client_id,
            "round": round_num,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy
        }
        self.client_step_records.append(record)

    def log_server_step_metrics(self, round_num: int, epoch: int, step: int, loss: float, accuracy: float = None, model_type: str = "student"):
        """Log detailed server training metrics for every step."""
        record = {
            "round": round_num,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "model_type": model_type
        }
        self.server_step_records.append(record)

    def log_cluster_evaluation(self, cluster_id: int, round_num: int, dataset: str, accuracy: float):
        """Log cluster model evaluation metrics."""
        record = {
            "cluster_id": cluster_id,
            "round": round_num,
            "dataset": dataset,
            "accuracy": accuracy
        }
        self.cluster_eval_records.append(record)

    def log_global_aggregation(self, round_num: int, cluster_id: int, num_clients: int, avg_loss: float, avg_accuracy: float):
        """Log global aggregation metrics for each cluster."""
        record = {
            "round": round_num,
            "cluster_id": cluster_id,
            "num_clients": num_clients,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy
        }
        self.global_aggregation_records.append(record)

    def log_cluster_aggregation_metrics(self, round_num: int, cluster_id: int, avg_loss: float, avg_accuracy: float, num_clients: int):
        """Log cluster aggregation average metrics for Excel storage."""
        record = {
            "round": round_num,
            "cluster_id": cluster_id,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "num_clients": num_clients
        }
        self.cluster_aggregation_records.append(record)
        # Save to Excel immediately after each cluster aggregation
        self._save_incremental_data()

    def log_client_epoch_metrics(self, client_id: int, cluster_id: int, round_num: int, epoch: int, loss: float, accuracy: float, epoch_time: float = None):
        """Log client epoch-level metrics immediately to Excel."""
        record = {
            "client_id": client_id,
            "cluster_id": cluster_id,
            "round": round_num,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "epoch_time": epoch_time
        }
        self.client_epoch_records.append(record)
        # Save to Excel immediately after each epoch
        self._save_incremental_data()

    def log_round_summary_metrics(self, round_num: int, cluster_summaries: List[Dict], global_summary: Dict = None):
        """Log round-level summary metrics after each global round."""
        for cluster_summary in cluster_summaries:
            record = {
                "round": round_num,
                "cluster_id": cluster_summary["cluster_id"],
                "num_clients": cluster_summary["num_clients"],
                "avg_loss": cluster_summary["avg_loss"],
                "avg_accuracy": cluster_summary["avg_accuracy"],
                "cluster_test_accuracy": cluster_summary.get("test_accuracy", None)
            }
            self.round_summary_records.append(record)
        
        # Save to Excel immediately after each round
        self._save_incremental_data()

    def log_posthoc_distillation_metrics(self, cluster_id: int, epoch: int, loss: float, accuracy: float, phase: str = "cluster_training"):
        """Log post-hoc knowledge distillation metrics to Excel."""
        record = {
            "cluster_id": cluster_id,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "phase": phase  # "cluster_training" or "student_distillation"
        }
        self.posthoc_distillation_records.append(record)
        # Save to Excel immediately after each post-hoc training step
        self._save_incremental_data()

    def log_client_training_time(self, client_id: int, cluster_id: int, round_num: int, total_time: float):
        """Log total client training time for a round."""
        record = {
            "type": "client_training",
            "client_id": client_id,
            "cluster_id": cluster_id,
            "round": round_num,
            "total_time": total_time
        }
        self.timing_records.append(record)
        self._save_incremental_data()

    def log_round_time(self, round_num: int, round_time: float):
        """Log total time for a complete round."""
        record = {
            "type": "round_total",
            "round": round_num,
            "total_time": round_time
        }
        self.timing_records.append(record)
        self._save_incremental_data()

    def log_cluster_training_time(self, cluster_id: int, round_num: int, training_time: float, phase: str = "distillation"):
        """Log cluster training time."""
        record = {
            "type": "cluster_training",
            "cluster_id": cluster_id,
            "round": round_num,
            "phase": phase,
            "training_time": training_time
        }
        self.timing_records.append(record)
        self._save_incremental_data()

    def log_overall_training_time(self, total_time: float):
        """Log overall training time for the entire federated learning process."""
        record = {
            "type": "overall_training",
            "total_time": total_time
        }
        self.timing_records.append(record)
        self._save_incremental_data()

    def _save_incremental_data(self, path: str = "metrics.xlsx"):
        """Save data incrementally to Excel file."""
        writer = pd.ExcelWriter(path, engine="openpyxl")
        
        # Removed client_epochs worksheet as requested
        # if self.client_epoch_records:
        #     pd.DataFrame(self.client_epoch_records).to_excel(writer, sheet_name="client_epochs", index=False)
        
        if self.round_summary_records:
            pd.DataFrame(self.round_summary_records).to_excel(writer, sheet_name="round_summaries", index=False)
        
        if self.local_records:
            pd.DataFrame(self.local_records).to_excel(writer, sheet_name="client_local", index=False)
        
        # Removed server_distill worksheet as requested
        # if self.distill_records:
        #     pd.DataFrame(self.distill_records).to_excel(writer, sheet_name="server_distill", index=False)
        
        if self.round_eval_records:
            pd.DataFrame(self.round_eval_records).to_excel(writer, sheet_name="round_eval", index=False)
        
        if self.client_step_records:
            pd.DataFrame(self.client_step_records).to_excel(writer, sheet_name="client_steps", index=False)
        
        if self.server_step_records:
            pd.DataFrame(self.server_step_records).to_excel(writer, sheet_name="server_steps", index=False)
        
        if self.cluster_eval_records:
            pd.DataFrame(self.cluster_eval_records).to_excel(writer, sheet_name="cluster_eval", index=False)
        
        # Removed global_aggregation worksheet as requested
        # if self.global_aggregation_records:
        #     pd.DataFrame(self.global_aggregation_records).to_excel(writer, sheet_name="global_aggregation", index=False)
        
        if self.cluster_aggregation_records:
            pd.DataFrame(self.cluster_aggregation_records).to_excel(writer, sheet_name="cluster_aggregation", index=False)
        
        if self.posthoc_distillation_records:
            pd.DataFrame(self.posthoc_distillation_records).to_excel(writer, sheet_name="posthoc_distillation", index=False)
        
        if self.timing_records:
            pd.DataFrame(self.timing_records).to_excel(writer, sheet_name="timing_data", index=False)
        
        writer.close()

    def save_excel(self, path: str = "metrics.xlsx"):
        writer = pd.ExcelWriter(path, engine="openpyxl")
        
        # Removed client_epochs worksheet as requested
        # if self.client_epoch_records:
        #     pd.DataFrame(self.client_epoch_records).to_excel(writer, sheet_name="client_epochs", index=False)
        
        if self.round_summary_records:
            pd.DataFrame(self.round_summary_records).to_excel(writer, sheet_name="round_summaries", index=False)
        
        if self.local_records:
            pd.DataFrame(self.local_records).to_excel(writer, sheet_name="client_local", index=False)
        
        # Removed server_distill worksheet as requested
        # if self.distill_records:
        #     pd.DataFrame(self.distill_records).to_excel(writer, sheet_name="server_distill", index=False)
        
        if self.round_eval_records:
            pd.DataFrame(self.round_eval_records).to_excel(writer, sheet_name="round_eval", index=False)
        
        if self.client_step_records:
            pd.DataFrame(self.client_step_records).to_excel(writer, sheet_name="client_steps", index=False)
        
        if self.server_step_records:
            pd.DataFrame(self.server_step_records).to_excel(writer, sheet_name="server_steps", index=False)
        
        if self.cluster_eval_records:
            pd.DataFrame(self.cluster_eval_records).to_excel(writer, sheet_name="cluster_eval", index=False)
        
        # Removed global_aggregation worksheet as requested
        # if self.global_aggregation_records:
        #     pd.DataFrame(self.global_aggregation_records).to_excel(writer, sheet_name="global_aggregation", index=False)
        
        if self.cluster_aggregation_records:
            pd.DataFrame(self.cluster_aggregation_records).to_excel(writer, sheet_name="cluster_aggregation", index=False)
        
        if self.posthoc_distillation_records:
            pd.DataFrame(self.posthoc_distillation_records).to_excel(writer, sheet_name="posthoc_distillation", index=False)
        
        writer.close()