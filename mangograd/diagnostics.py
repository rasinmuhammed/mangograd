import numpy as np

def gradient_health(model):
    """
    Print a per-parameter health report of gradient magnitudes.
    Call this after loss.backward() to see if your network is healthy.
    """
    print("\n🥭 Mangograd Gradient Health Report")
    print("=" * 70)
    
    for i, p in enumerate(model.parameters()):
        grad_abs = np.abs(p.grad)
        mean_grad = grad_abs.mean()
        max_grad = grad_abs.max()
        
        # Check for dead parameters (gradient is exactly zero everywhere)
        pct_zero = (p.grad == 0).mean() * 100
        
        # Determine health status
        if np.any(np.isnan(p.grad)):
            status = "💀 NaN"
        elif max_grad > 1000:
            status = "🔥 EXPLODING"
        elif mean_grad < 1e-7:
            status = "🧊 VANISHING"
        elif pct_zero > 50:
            status = f"💤 {pct_zero:.0f}% DEAD"
        else:
            status = "✅ HEALTHY"
        
        print(f"  param_{i} | shape: {str(p.data.shape):<14} | "
              f"grad mean: {mean_grad:.2e} | "
              f"grad max: {max_grad:.2e} | "
              f"zeros: {pct_zero:5.1f}% | {status}")
    
    print("=" * 70)
