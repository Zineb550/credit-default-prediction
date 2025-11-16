"""
Simple Monitoring Dashboard
Visualizes API predictions and performance
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

BASE_URL = "http://localhost:8000"


def get_stats():
    """Fetch current statistics from API"""
    try:
        response = requests.get(f"{BASE_URL}/stats")
        return response.json()
    except:
        return None


def get_drift():
    """Fetch drift information"""
    try:
        response = requests.get(f"{BASE_URL}/monitoring/drift")
        return response.json()
    except:
        return None


def create_dashboard():
    """Create monitoring dashboard"""
    print("\n" + "="*80)
    print("📊 CREDIT DEFAULT PREDICTION API - MONITORING DASHBOARD")
    print("="*80)
    
    # Get stats
    stats = get_stats()
    
    if stats is None:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the API is running: python app/api.py")
        return
    
    if stats.get('total_predictions', 0) == 0:
        print("\n⚠️  No predictions yet!")
        print("Make some predictions first using: python app/test_api.py")
        return
    
    # Display stats
    print(f"\n📈 PREDICTION STATISTICS")
    print("-" * 80)
    print(f"Total Predictions:       {stats.get('total_predictions', 0)}")
    print(f"Default Predictions:     {stats.get('default_predictions', 0)}")
    print(f"Default Rate:            {stats.get('default_rate', 0):.2%}")
    print(f"Average Probability:     {stats.get('average_probability', 0):.4f}")
    print(f"Min Probability:         {stats.get('min_probability', 0):.4f}")
    print(f"Max Probability:         {stats.get('max_probability', 0):.4f}")
    
    # Get drift info
    drift = get_drift()
    
    if drift and 'message' not in drift:
        print(f"\n🔍 DATA DRIFT ANALYSIS")
        print("-" * 80)
        print(f"Expected Default Rate:   {drift.get('expected_default_rate', 0):.2%}")
        print(f"Recent Default Rate:     {drift.get('recent_default_rate', 0):.2%}")
        print(f"Drift Amount:            {drift.get('drift', 0):.2%}")
        print(f"Sample Size:             {drift.get('sample_size', 0)}")
        
        status = drift.get('status', 'Unknown')
        alert = drift.get('alert', False)
        
        if alert:
            print(f"Status:                  ⚠️  {status}")
        else:
            print(f"Status:                  ✅ {status}")
    
    print("\n" + "="*80)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def monitor_continuous(interval=5):
    """Continuous monitoring - updates every interval seconds"""
    print(f"\n🔄 Starting continuous monitoring (updates every {interval}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            create_dashboard()
            time.sleep(interval)
            print("\n" + "↻ Refreshing..." + "\n")
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")


def plot_statistics():
    """Plot prediction statistics"""
    stats = get_stats()
    
    if stats is None or stats.get('total_predictions', 0) == 0:
        print("No data to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Default vs No Default
    ax1 = axes[0]
    categories = ['No Default', 'Default']
    values = [
        stats['total_predictions'] - stats['default_predictions'],
        stats['default_predictions']
    ]
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.bar(categories, values, color=colors)
    ax1.set_title('Prediction Distribution', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Count')
    
    # Add value labels
    for i, v in enumerate(values):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Probability range
    ax2 = axes[1]
    prob_data = {
        'Average': stats['average_probability'],
        'Min': stats['min_probability'],
        'Max': stats['max_probability']
    }
    
    ax2.bar(prob_data.keys(), prob_data.values(), color='#3498db')
    ax2.set_title('Probability Statistics', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for i, (k, v) in enumerate(prob_data.items()):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"monitoring_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Dashboard plot saved: {filename}")
    
    plt.show()


def main():
    """Main function"""
    import sys
    
    print("\n🎯 MONITORING OPTIONS:")
    print("1. Show current dashboard (one-time)")
    print("2. Continuous monitoring (updates every 5s)")
    print("3. Generate plot visualization")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            create_dashboard()
        elif choice == "2":
            monitor_continuous(interval=5)
        elif choice == "3":
            plot_statistics()
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\n✅ Exiting...")


if __name__ == "__main__":
    main()
