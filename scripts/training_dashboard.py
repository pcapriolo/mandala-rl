#!/usr/bin/env python3
"""
Training Dashboard for Mandala RL

Displays all key metrics, their explanations, and health indicators.
"""
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  Tensorboard not available - install with: pip install tensorboard")


class TrainingDashboard:
    """Comprehensive training metrics dashboard."""

    def __init__(self, log_dir="data/logs", checkpoint_dir="data/checkpoints",
                 elo_file="data/elo_ratings.json"):
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.elo_file = Path(elo_file)

        # Health thresholds
        self.thresholds = {
            'loss': {'good': 4.0, 'warning': 5.0, 'bad': 6.0},
            'policy_loss': {'good': 2.5, 'warning': 3.5, 'bad': 4.5},
            'value_loss': {'good': 1.0, 'warning': 1.5, 'bad': 2.0},
            'elo_change': {'good': 20, 'warning': 5, 'bad': -10},
            'win_rate': {'good': 0.55, 'warning': 0.45, 'bad': 0.35},
        }

    def get_checkpoint_info(self):
        """Get info from latest checkpoint."""
        latest = self.checkpoint_dir / "model_latest.pt"
        if not latest.exists():
            return None

        try:
            checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            return {
                'iteration': checkpoint.get('iteration', 0),
                'total_games': checkpoint.get('total_games', 0),
                'buffer_size': len(checkpoint.get('replay_buffer', [])),
                'timestamp': datetime.fromtimestamp(latest.stat().st_mtime),
            }
        except Exception as e:
            return {'error': str(e)}

    def get_tensorboard_metrics(self):
        """Extract metrics from Tensorboard logs."""
        if not TENSORBOARD_AVAILABLE:
            return None

        # Find most recent event file
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            return None

        latest_event = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]

        try:
            ea = event_accumulator.EventAccumulator(str(latest_event))
            ea.Reload()

            metrics = {}

            # Get scalars
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                if events:
                    # Get last 10 values for trend analysis
                    recent = events[-10:] if len(events) >= 10 else events
                    metrics[tag] = {
                        'current': events[-1].value,
                        'recent': [e.value for e in recent],
                        'step': events[-1].step,
                    }

            return metrics
        except Exception as e:
            return {'error': str(e)}

    def parse_loss_from_checkpoints(self):
        """Get loss history from checkpoint files."""
        checkpoints = sorted(self.checkpoint_dir.glob("model_iter_*.pt"))

        if not checkpoints:
            return None

        loss_history = []

        for ckpt_path in checkpoints:
            try:
                # Extract iteration number from filename
                iter_num = int(ckpt_path.stem.split('_')[-1])

                # We don't have loss in checkpoint, but we can note the iteration
                loss_history.append({'iteration': iter_num})
            except Exception:
                continue

        return loss_history if loss_history else None

    def get_elo_info(self):
        """Get Elo rating information."""
        if not self.elo_file.exists():
            return None

        try:
            with open(self.elo_file, 'r') as f:
                elo_data = json.load(f)

            # Get ratings sorted by iteration
            ratings = []
            for model_id, rating in elo_data.items():
                if model_id.startswith('iter_'):
                    iter_num = int(model_id.split('_')[1])
                    ratings.append((iter_num, rating))

            ratings.sort()

            if len(ratings) >= 2:
                current_rating = ratings[-1][1]
                prev_rating = ratings[-2][1]
                elo_change = current_rating - prev_rating
            else:
                current_rating = ratings[-1][1] if ratings else 1500
                elo_change = 0

            return {
                'current': current_rating,
                'change': elo_change,
                'history': ratings,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_health_status(self, value, metric_type):
        """Get health status (good/warning/bad) for a metric."""
        if metric_type not in self.thresholds:
            return 'unknown', '⚪'

        t = self.thresholds[metric_type]

        # For metrics where lower is better (loss)
        if metric_type in ['loss', 'policy_loss', 'value_loss']:
            if value <= t['good']:
                return 'good', '🟢'
            elif value <= t['warning']:
                return 'warning', '🟡'
            else:
                return 'bad', '🔴'

        # For metrics where higher is better (elo_change, win_rate)
        else:
            if value >= t['good']:
                return 'good', '🟢'
            elif value >= t['warning']:
                return 'warning', '🟡'
            else:
                return 'bad', '🔴'

    def calculate_trend(self, values):
        """Calculate trend direction from recent values."""
        if len(values) < 2:
            return 'stable', '→'

        # Simple linear trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable', '→'

        slope = numerator / denominator

        # Relative to mean
        rel_slope = slope / (abs(y_mean) + 1e-8)

        if abs(rel_slope) < 0.01:
            return 'stable', '→'
        elif rel_slope < 0:
            return 'decreasing', '↓'
        else:
            return 'increasing', '↑'

    def print_metric(self, name, value, unit='', description='',
                    good='', warning='', bad='', health_type=None, trend_values=None):
        """Print a metric with health indicator and explanation."""
        print(f"\n{'─' * 70}")
        print(f"📊 {name}")
        print(f"{'─' * 70}")

        # Current value with health indicator
        if health_type:
            status, emoji = self.get_health_status(value, health_type)
            print(f"Current: {emoji} {value:.4f}{unit}")
        else:
            print(f"Current: {value:.4f}{unit}")

        # Trend
        if trend_values:
            trend_dir, arrow = self.calculate_trend(trend_values)
            print(f"Trend:   {arrow} {trend_dir.capitalize()} (last 10 values)")

        # Description
        if description:
            print(f"\n💡 What it means:")
            print(f"   {description}")

        # Health indicators
        if good or warning or bad:
            print(f"\n📈 What to watch for:")
            if good:
                print(f"   🟢 Good:    {good}")
            if warning:
                print(f"   🟡 Warning: {warning}")
            if bad:
                print(f"   🔴 Bad:     {bad}")

    def display(self):
        """Display complete dashboard."""
        print("=" * 70)
        print("🎯 MANDALA RL TRAINING DASHBOARD")
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Checkpoint Info
        print("\n" + "=" * 70)
        print("📦 CHECKPOINT STATUS")
        print("=" * 70)

        checkpoint_info = self.get_checkpoint_info()
        if checkpoint_info and 'error' not in checkpoint_info:
            print(f"Iteration:     {checkpoint_info['iteration']}")
            print(f"Total Games:   {checkpoint_info['total_games']:,}")
            print(f"Buffer Size:   {checkpoint_info['buffer_size']:,} examples")
            print(f"Last Updated:  {checkpoint_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

            current_iter = checkpoint_info['iteration']
            buffer_size = checkpoint_info['buffer_size']
            total_games = checkpoint_info['total_games']
        else:
            print("⚠️  No checkpoint found or error loading")
            current_iter = 0
            buffer_size = 0
            total_games = 0

        # 2. Loss Metrics
        print("\n" + "=" * 70)
        print("📉 LOSS METRICS")
        print("=" * 70)

        tb_metrics = self.get_tensorboard_metrics()
        if tb_metrics and 'error' not in tb_metrics:
            # Total Loss
            if 'Training/TotalLoss' in tb_metrics:
                loss_data = tb_metrics['Training/TotalLoss']
                self.print_metric(
                    name="Total Loss",
                    value=loss_data['current'],
                    description="Combined policy + value loss. Overall measure of learning progress.",
                    good="< 4.0 (well-trained)",
                    warning="4.0 - 5.0 (still learning)",
                    bad="> 5.0 (early training or issues)",
                    health_type='loss',
                    trend_values=loss_data['recent']
                )

            # Policy Loss
            if 'Training/PolicyLoss' in tb_metrics:
                policy_data = tb_metrics['Training/PolicyLoss']
                self.print_metric(
                    name="Policy Loss (Cross-Entropy)",
                    value=policy_data['current'],
                    description="How well network predicts MCTS move probabilities. Lower = better move selection.",
                    good="< 2.5 (confident predictions)",
                    warning="2.5 - 3.5 (learning)",
                    bad="> 3.5 (uncertain or exploring)",
                    health_type='policy_loss',
                    trend_values=policy_data['recent']
                )

            # Value Loss
            if 'Training/ValueLoss' in tb_metrics:
                value_data = tb_metrics['Training/ValueLoss']
                self.print_metric(
                    name="Value Loss (MSE)",
                    value=value_data['current'],
                    description="How well network predicts game outcomes. Lower = better position evaluation.",
                    good="< 1.0 (accurate predictions)",
                    warning="1.0 - 1.5 (decent accuracy)",
                    bad="> 1.5 (poor predictions)",
                    health_type='value_loss',
                    trend_values=value_data['recent']
                )
        else:
            print("⚠️  No Tensorboard metrics available")
            if not TENSORBOARD_AVAILABLE:
                print("   Install with: pip install tensorboard")

        # 3. Elo Ratings
        print("\n" + "=" * 70)
        print("🏆 ELO RATINGS")
        print("=" * 70)

        elo_info = self.get_elo_info()
        if elo_info and 'error' not in elo_info:
            self.print_metric(
                name="Current Elo Rating",
                value=elo_info['current'],
                description="Relative playing strength vs all previous versions. Most important metric!",
                good="> +20 per iteration (rapid improvement)",
                warning="+5 to +20 per iteration (steady progress)",
                bad="< +5 or negative (plateau or regression)",
                health_type=None  # No auto health check for absolute Elo
            )

            if elo_info['change'] != 0:
                status, emoji = self.get_health_status(elo_info['change'], 'elo_change')
                print(f"\n{emoji} Elo Change: {elo_info['change']:+.1f} points")

            # Show history
            if len(elo_info['history']) > 1:
                print(f"\n📈 Recent History:")
                for iter_num, rating in elo_info['history'][-5:]:
                    print(f"   Iteration {iter_num:2d}: {rating:7.1f}")
        else:
            print("⚠️  No Elo data available (run evaluation first)")

        # 4. Training Efficiency
        print("\n" + "=" * 70)
        print("⚡ TRAINING EFFICIENCY")
        print("=" * 70)

        if current_iter > 0:
            examples_per_iter = buffer_size / max(current_iter, 1)
            games_per_iter = total_games / max(current_iter, 1)

            print(f"Examples per Iteration: {examples_per_iter:,.0f}")
            print(f"Games per Iteration:    {games_per_iter:,.0f}")

            print(f"\n💡 What it means:")
            print(f"   Each iteration adds ~{examples_per_iter:.0f} training examples")
            print(f"   from ~{games_per_iter:.0f} self-play games.")

            print(f"\n📈 What to watch for:")
            print(f"   🟢 Good:    Consistent growth (±10%)")
            print(f"   🟡 Warning: Large fluctuations (>25%)")
            print(f"   🔴 Bad:     Decreasing trend")

        # 5. Data Statistics
        print("\n" + "=" * 70)
        print("💾 REPLAY BUFFER")
        print("=" * 70)

        print(f"Total Examples:  {buffer_size:,}")
        print(f"Total Games:     {total_games:,}")
        if total_games > 0:
            avg_examples = buffer_size / total_games
            print(f"Examples/Game:   {avg_examples:.1f}")

        print(f"\n💡 What it means:")
        print(f"   Larger buffer = more diverse training data")
        print(f"   Target: 50k-500k examples depending on game complexity")

        print(f"\n📈 What to watch for:")
        print(f"   🟢 Good:    Steady growth, high diversity")
        print(f"   🟡 Warning: Plateau or very slow growth")
        print(f"   🔴 Bad:     Decreasing size")

        # 6. Overall Health Check
        print("\n" + "=" * 70)
        print("🏥 OVERALL HEALTH CHECK")
        print("=" * 70)

        health_items = []

        # Check loss
        if tb_metrics and 'Training/TotalLoss' in tb_metrics:
            loss = tb_metrics['Training/TotalLoss']['current']
            status, emoji = self.get_health_status(loss, 'loss')
            health_items.append((emoji, f"Total Loss: {loss:.3f}", status))

        # Check Elo
        if elo_info and elo_info.get('change', 0) != 0:
            change = elo_info['change']
            status, emoji = self.get_health_status(change, 'elo_change')
            health_items.append((emoji, f"Elo Change: {change:+.1f}", status))

        # Check buffer growth
        if current_iter >= 2:
            if buffer_size > 20000:
                health_items.append(("🟢", f"Buffer Size: {buffer_size:,}", "good"))
            elif buffer_size > 10000:
                health_items.append(("🟡", f"Buffer Size: {buffer_size:,}", "warning"))
            else:
                health_items.append(("🔴", f"Buffer Size: {buffer_size:,}", "bad"))

        if health_items:
            good_count = sum(1 for _, _, s in health_items if s == 'good')
            total_count = len(health_items)

            print(f"\nHealth Score: {good_count}/{total_count} metrics in good range")
            print()
            for emoji, text, status in health_items:
                print(f"  {emoji} {text}")

            if good_count == total_count:
                print(f"\n✅ All systems healthy! Training is progressing well.")
            elif good_count >= total_count * 0.6:
                print(f"\n⚠️  Mostly healthy, but watch warning indicators.")
            else:
                print(f"\n🔴 Multiple issues detected. Review metrics above.")
        else:
            print("⚠️  Not enough data for health assessment")

        # 7. Recommendations
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS")
        print("=" * 70)

        recommendations = []

        # Loss-based recommendations
        if tb_metrics and 'Training/TotalLoss' in tb_metrics:
            loss = tb_metrics['Training/TotalLoss']['current']
            recent = tb_metrics['Training/TotalLoss']['recent']
            trend, _ = self.calculate_trend(recent)

            if loss > 5.0 and trend == 'increasing':
                recommendations.append("⚠️  Loss increasing - check for training instability")
            elif trend == 'stable' and loss > 3.0 and current_iter > 20:
                recommendations.append("💡 Loss plateaued - consider adjusting learning rate")

        # Elo-based recommendations
        if elo_info and len(elo_info.get('history', [])) >= 3:
            recent_changes = []
            history = elo_info['history']
            for i in range(len(history) - 1):
                recent_changes.append(history[i+1][1] - history[i][1])

            if recent_changes and all(c < 5 for c in recent_changes[-3:]):
                recommendations.append("⚠️  Elo not improving - training may have converged")

        # Buffer recommendations
        if buffer_size < 20000 and current_iter > 5:
            recommendations.append("💡 Consider increasing games per iteration for more data diversity")

        # Iteration recommendations
        if current_iter < 10:
            recommendations.append("📊 Early training - expect high loss and exploration")
        elif current_iter >= 50:
            recommendations.append("🎯 Mid-late training - focus on Elo improvements")

        if recommendations:
            for rec in recommendations:
                print(f"\n{rec}")
        else:
            print("\n✅ Training looks good! Continue current configuration.")

        print("\n" + "=" * 70)
        print("Dashboard complete! Run again to see updated metrics.")
        print("=" * 70)


def main():
    """Run the dashboard."""
    dashboard = TrainingDashboard()
    dashboard.display()


if __name__ == "__main__":
    main()
