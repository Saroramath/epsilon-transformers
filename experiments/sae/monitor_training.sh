#!/bin/bash
# Monitor SAE training progress

echo "SAE Training Monitor"
echo "===================="
echo ""

# Check if tmux session exists
if tmux has-session -t sae_training 2>/dev/null; then
    echo "✓ Training session is active"

    # Show last 20 lines of log
    echo ""
    echo "Recent training output:"
    echo "----------------------"
    tail -20 sae_training.log

    echo ""
    echo "Commands:"
    echo "  tail -f sae_training.log    - Watch live updates"
    echo "  tmux attach -t sae_training - Attach to training session"
    echo "  ./monitor_training.sh       - Refresh this view"
else
    echo "✗ Training session not found"
    echo ""
    echo "To start training:"
    echo "  tmux new-session -d -s sae_training \"source ../../.venv/bin/activate && python train_sae.py 2>&1 | tee sae_training.log\""
fi
