#!/bin/bash
# Experiment runner script
# Usage: ./run_experiments.sh [experiment_name]

set -e

EXPERIMENT=${1:-cosine_lr}
LOG_DIR="mainrun/logs"
RESULTS_FILE="experiment_results.json"

echo "🚀 Starting experiment: $EXPERIMENT"
echo "📊 Results will be saved to $RESULTS_FILE"
echo "📝 Logs will be saved to $LOG_DIR/"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p checkpoints

# Function to run experiment
run_experiment() {
    local name=$1
    local description=$2
    
    echo ""
    echo "═" 60
    echo "🧪 Running: $name"
    echo "📝 $description"
    echo "═" 60
    
    # Create experiment branch
    git checkout -b "experiment-$name" 2>/dev/null || git checkout "experiment-$name"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the experiment based on type
    case $name in
        "cosine_lr")
            python3 cosine_experiment.py
            ;;
        "wider_model")
            python3 experiments.py v10_wider_model
            ;;
        "deeper_model")
            python3 experiments.py v11_deeper_model
            ;;
        "adaptive_dropout")
            python3 experiments.py v14_adaptive_dropout
            ;;
        "token_augment")
            python3 experiments.py v13_data_augment
            ;;
        *)
            echo "❌ Unknown experiment: $name"
            return 1
            ;;
    esac
    
    # Record end time and duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "⏱️ Experiment completed in ${duration} seconds"
    
    # Extract best loss if available
    if [ -f "$LOG_DIR/mainrun.log" ]; then
        best_loss=$(grep '"event": "validation_step"' "$LOG_DIR/mainrun.log" | jq -s 'min_by(.loss) | .loss' 2>/dev/null || echo "null")
        echo "📈 Best validation loss: $best_loss"
        
        # Record result
        result=$(cat <<EOF
{
  "name": "$name",
  "description": "$description",
  "best_loss": $best_loss,
  "duration_seconds": $duration,
  "timestamp": "$(date -Iseconds)",
  "status": "completed"
}
EOF
        )
        
        # Add to results file
        if [ -f "$RESULTS_FILE" ]; then
            temp=$(mktemp)
            jq ". += [$result]" "$RESULTS_FILE" > "$temp" && mv "$temp" "$RESULTS_FILE"
        else
            echo "[$result]" > "$RESULTS_FILE"
        fi
        
        # Compare with baseline
        baseline=0.974555
        if [[ "$best_loss" != "null" ]]; then
            improvement=$(echo "scale=6; ($baseline - $best_loss) / $baseline * 100" | bc -l 2>/dev/null || echo "0")
            echo "🎯 Improvement over baseline: ${improvement}%"
            
            if (( $(echo "$best_loss < $baseline" | bc -l) )); then
                echo "🎉 NEW BEST RESULT!"
            fi
        fi
    fi
    
    # Return to main branch
    git checkout main
    
    echo ""
    echo "✅ Experiment $name completed"
}

# Main execution
case $EXPERIMENT in
    "all")
        echo "🔄 Running all experiments sequentially..."
        run_experiment "cosine_lr" "Cosine learning rate with warmup"
        run_experiment "wider_model" "Wider model (d_model=768, 6 layers)"
        run_experiment "deeper_model" "Deeper model (d_model=512, 10 layers)"
        run_experiment "adaptive_dropout" "Adaptive dropout scheduling"
        run_experiment "token_augment" "Token-level data augmentation"
        ;;
    "quick")
        echo "⚡ Running quick test (cosine LR only)..."
        run_experiment "cosine_lr" "Cosine learning rate with warmup"
        ;;
    *)
        if [[ -f "cosine_experiment.py" ]] && [[ "$EXPERIMENT" == "cosine_lr" ]]; then
            run_experiment "cosine_lr" "Cosine learning rate with warmup"
        elif [[ -f "experiments.py" ]]; then
            run_experiment "$EXPERIMENT" "Custom experiment"
        else
            echo "❌ Unknown experiment: $EXPERIMENT"
            echo "Available: cosine_lr, wider_model, deeper_model, adaptive_dropout, token_augment, all, quick"
            exit 1
        fi
        ;;
esac

echo ""
echo "🏁 All experiments completed!"
echo "📊 Results summary:"
if [ -f "$RESULTS_FILE" ]; then
    jq -r '.[] | "• \(.name): \(.best_loss // "N/A") (\(.duration_seconds // "N/A")s)"' "$RESULTS_FILE"
else
    echo "No results recorded"
fi