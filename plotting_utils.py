import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FedMTFIPlotter:
    def __init__(self, output_dir: str = "plots"):
        """Initialize plotter with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_metrics_from_excel(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """Load all metrics from Excel file."""
        metrics = {}
        try:
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(excel_path)
            for sheet_name in excel_file.sheet_names:
                metrics[sheet_name] = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"[Plotter] Loaded metrics from {excel_path}")
            return metrics
        except Exception as e:
            print(f"[Plotter] Error loading metrics: {e}")
            return {}
    
    def plot_client_training_progress(self, client_steps_df: pd.DataFrame):
        """Plot client training progress over steps."""
        if client_steps_df.empty:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot loss over steps for each client
        plt.subplot(2, 2, 1)
        for client_id in client_steps_df['client_id'].unique():
            client_data = client_steps_df[client_steps_df['client_id'] == client_id]
            plt.plot(client_data.index, client_data['loss'], label=f'Client {client_id}', alpha=0.7)
        plt.title('Client Training Loss Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy over steps for each client
        plt.subplot(2, 2, 2)
        for client_id in client_steps_df['client_id'].unique():
            client_data = client_steps_df[client_steps_df['client_id'] == client_id]
            if 'accuracy' in client_data.columns and not client_data['accuracy'].isna().all():
                plt.plot(client_data.index, client_data['accuracy'], label=f'Client {client_id}', alpha=0.7)
        plt.title('Client Training Accuracy Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot loss by round and epoch
        plt.subplot(2, 2, 3)
        round_loss = client_steps_df.groupby(['round', 'epoch'])['loss'].mean().reset_index()
        for round_num in round_loss['round'].unique():
            round_data = round_loss[round_loss['round'] == round_num]
            plt.plot(round_data['epoch'], round_data['loss'], marker='o', label=f'Round {round_num}')
        plt.title('Average Loss by Round and Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy by round and epoch
        plt.subplot(2, 2, 4)
        if 'accuracy' in client_steps_df.columns and not client_steps_df['accuracy'].isna().all():
            round_acc = client_steps_df.groupby(['round', 'epoch'])['accuracy'].mean().reset_index()
            for round_num in round_acc['round'].unique():
                round_data = round_acc[round_acc['round'] == round_num]
                plt.plot(round_data['epoch'], round_data['accuracy'], marker='o', label=f'Round {round_num}')
            plt.title('Average Accuracy by Round and Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Average Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'client_training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved client training progress plot")
    
    def plot_server_training_progress(self, server_steps_df: pd.DataFrame):
        """Plot server training progress over steps."""
        if server_steps_df.empty:
            return
            
        plt.figure(figsize=(15, 8))
        
        # Plot loss over steps for different model types
        plt.subplot(2, 2, 1)
        for model_type in server_steps_df['model_type'].unique():
            model_data = server_steps_df[server_steps_df['model_type'] == model_type]
            plt.plot(model_data.index, model_data['loss'], label=f'{model_type}', alpha=0.7)
        plt.title('Server Training Loss Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy over steps for different model types
        plt.subplot(2, 2, 2)
        for model_type in server_steps_df['model_type'].unique():
            model_data = server_steps_df[server_steps_df['model_type'] == model_type]
            if 'accuracy' in model_data.columns and not model_data['accuracy'].isna().all():
                plt.plot(model_data.index, model_data['accuracy'], label=f'{model_type}', alpha=0.7)
        plt.title('Server Training Accuracy Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss by round and epoch
        plt.subplot(2, 2, 3)
        round_loss = server_steps_df.groupby(['round', 'epoch'])['loss'].mean().reset_index()
        plt.plot(round_loss.index, round_loss['loss'], marker='o', color='red')
        plt.title('Server Average Loss by Training Step')
        plt.xlabel('Training Step')
        plt.ylabel('Average Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy by round and epoch
        plt.subplot(2, 2, 4)
        if 'accuracy' in server_steps_df.columns and not server_steps_df['accuracy'].isna().all():
            round_acc = server_steps_df.groupby(['round', 'epoch'])['accuracy'].mean().reset_index()
            plt.plot(round_acc.index, round_acc['accuracy'], marker='o', color='blue')
            plt.title('Server Average Accuracy by Training Step')
            plt.xlabel('Training Step')
            plt.ylabel('Average Accuracy')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'server_training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved server training progress plot")
    
    def plot_cluster_evaluation(self, cluster_eval_df: pd.DataFrame):
        """Plot cluster evaluation metrics."""
        if cluster_eval_df.empty:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot cluster accuracy over rounds
        plt.subplot(2, 2, 1)
        for cluster_id in cluster_eval_df['cluster_id'].unique():
            cluster_data = cluster_eval_df[cluster_eval_df['cluster_id'] == cluster_id]
            plt.plot(cluster_data['round'], cluster_data['accuracy'], marker='o', label=f'Cluster {cluster_id}')
        plt.title('Cluster Accuracy Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot final cluster accuracies
        plt.subplot(2, 2, 2)
        final_round = cluster_eval_df['round'].max()
        final_accuracies = cluster_eval_df[cluster_eval_df['round'] == final_round]
        plt.bar(final_accuracies['cluster_id'].astype(str), final_accuracies['accuracy'])
        plt.title(f'Final Cluster Accuracies (Round {final_round})')
        plt.xlabel('Cluster ID')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy distribution by cluster
        plt.subplot(2, 2, 3)
        cluster_eval_df.boxplot(column='accuracy', by='cluster_id', ax=plt.gca())
        plt.title('Accuracy Distribution by Cluster')
        plt.suptitle('')  # Remove default title
        plt.xlabel('Cluster ID')
        plt.ylabel('Accuracy')
        
        # Plot accuracy improvement over rounds
        plt.subplot(2, 2, 4)
        avg_accuracy_by_round = cluster_eval_df.groupby('round')['accuracy'].mean()
        plt.plot(avg_accuracy_by_round.index, avg_accuracy_by_round.values, marker='o', color='green', linewidth=2)
        plt.title('Average Cluster Accuracy Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Average Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved cluster evaluation plot")
    
    def plot_global_aggregation(self, global_agg_df: pd.DataFrame):
        """Plot global aggregation metrics."""
        if global_agg_df.empty:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot average loss by cluster over rounds
        plt.subplot(2, 2, 1)
        for cluster_id in global_agg_df['cluster_id'].unique():
            cluster_data = global_agg_df[global_agg_df['cluster_id'] == cluster_id]
            plt.plot(cluster_data['round'], cluster_data['avg_loss'], marker='o', label=f'Cluster {cluster_id}')
        plt.title('Average Loss by Cluster Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot average accuracy by cluster over rounds
        plt.subplot(2, 2, 2)
        for cluster_id in global_agg_df['cluster_id'].unique():
            cluster_data = global_agg_df[global_agg_df['cluster_id'] == cluster_id]
            plt.plot(cluster_data['round'], cluster_data['avg_accuracy'], marker='o', label=f'Cluster {cluster_id}')
        plt.title('Average Accuracy by Cluster Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Average Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot number of clients per cluster
        plt.subplot(2, 2, 3)
        cluster_sizes = global_agg_df.groupby('cluster_id')['num_clients'].first()
        plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values)
        plt.title('Number of Clients per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Clients')
        plt.grid(True, alpha=0.3)
        
        # Plot overall training progress
        plt.subplot(2, 2, 4)
        overall_progress = global_agg_df.groupby('round')[['avg_loss', 'avg_accuracy']].mean()
        ax1 = plt.gca()
        ax1.plot(overall_progress.index, overall_progress['avg_loss'], 'r-', marker='o', label='Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Average Loss', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        ax2 = ax1.twinx()
        ax2.plot(overall_progress.index, overall_progress['avg_accuracy'], 'b-', marker='s', label='Accuracy')
        ax2.set_ylabel('Average Accuracy', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title('Overall Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'global_aggregation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved global aggregation plot")
    
    def plot_round_evaluation(self, round_eval_df: pd.DataFrame):
        """Plot round evaluation metrics."""
        if round_eval_df.empty:
            return
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(round_eval_df['round'], round_eval_df['test_acc'], marker='o', linewidth=2, markersize=6)
        plt.title('Test Accuracy Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(round_eval_df['round'], round_eval_df['test_acc'], 1)
        p = np.poly1d(z)
        plt.plot(round_eval_df['round'], p(round_eval_df['round']), "r--", alpha=0.8, label='Trend')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'round_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved round evaluation plot")
    
    def create_summary_report(self, metrics: Dict[str, pd.DataFrame]):
        """Create a comprehensive summary report with all plots."""
        print(f"[Plotter] Creating comprehensive summary report...")
        
        # Plot client training progress
        if 'client_steps' in metrics:
            self.plot_client_training_progress(metrics['client_steps'])
        
        # Plot server training progress
        if 'server_steps' in metrics:
            self.plot_server_training_progress(metrics['server_steps'])
        
        # Plot cluster evaluation
        if 'cluster_eval' in metrics:
            self.plot_cluster_evaluation(metrics['cluster_eval'])
        
        # Plot global aggregation
        if 'global_aggregation' in metrics:
            self.plot_global_aggregation(metrics['global_aggregation'])
        
        # Plot round evaluation
        if 'round_eval' in metrics:
            self.plot_round_evaluation(metrics['round_eval'])
        
        print(f"[Plotter] Summary report completed. All plots saved in '{self.output_dir}' folder.")


def create_plots_from_excel(excel_path: str, output_dir: str = "plots"):
    """Convenience function to create all plots from Excel file."""
    plotter = FedMTFIPlotter(output_dir)
    metrics = plotter.load_metrics_from_excel(excel_path)
    plotter.create_summary_report(metrics)
    return plotter