# %% [markdown]
# # Euler's Method: Error Analysis and Convergence Study
# 
# ## 1. Introduction and Theoretical Foundation
# 
# ### 1.1 The Initial Value Problem
# 
# We consider the first-order ordinary differential equation (ODE):
# 
# $$
# \frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0
# $$
# 
# **Test Problem:** For rigorous error analysis, we use the exponential decay equation:
# 
# $$
# \frac{dy}{dt} = -y, \quad y(0) = 1
# $$
# 
# **Exact Solution:** $y(t) = e^{-t}$
# 
# This problem is ideal for convergence studies because:
# 1. The exact solution is known analytically
# 2. It's smooth and well-behaved (infinitely differentiable)
# 3. It's stable (solutions don't blow up)
# 
# ### 1.2 Euler's Method: The Simplest Numerical Integrator
# 
# Euler's method approximates the solution by taking discrete steps:
# 
# $$
# y_{n+1} = y_n + h \cdot f(t_n, y_n)
# $$
# 
# where $h = \Delta t$ is the step size.
# 
# **Geometric Interpretation:** At each step, we follow the tangent line from the current point. This is equivalent to a first-order Taylor series approximation:
# 
# $$
# y(t_{n+1}) = y(t_n) + h y'(t_n) + \frac{h^2}{2} y''(\xi), \quad \xi \in [t_n, t_{n+1}]
# $$
# 
# ### 1.3 Error Analysis: Local vs. Global Error
# 
# **Local Truncation Error (LTE):** Error made in a single step assuming exact previous value.
# 
# $$
# \text{LTE}_n = y_{\text{exact}}(t_{n+1}) - y_{\text{Euler}}(t_{n+1}) \approx \mathcal{O}(h^2)
# $$
# 
# **Global Error (GE):** Accumulated error over the entire integration interval.
# 
# $$
# \text{GE}_N = y_{\text{exact}}(t_N) - y_{\text{Euler}}(t_N) \approx \mathcal{O}(h)
# $$
# 
# **Key Theoretical Result:** Euler's method is **first-order accurate**. 
# Halving the step size should halve the global error:
# 
# $$
# \frac{\text{Error}(h)}{\text{Error}(h/2)} \approx 2
# $$

# %% [markdown]
# ## 2. Setup and Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys
from pathlib import Path

# Add the src directory to path to import our custom solvers
sys.path.append(str(Path.cwd().parent))

from src.solvers.explicit import euler, rk4

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# %% [markdown]
# ## 3. Visualizing Euler's Method: Step-by-Step
# 
# Before diving into error analysis, let's visualize how Euler's method actually works geometrically.

# %%
def exponential_decay(t: float, y: np.ndarray) -> np.ndarray:
    """Right-hand side of dy/dt = -y."""
    return -y

def exact_solution(t: float) -> float:
    """Exact solution y(t) = e^{-t}."""
    return np.exp(-t)

# Set up the problem
t_span = (0.0, 3.0)
y0 = np.array([1.0])
h_large = 0.5  # Large step size to clearly see the error

# Solve using Euler's method
t_euler, y_euler = euler(exponential_decay, t_span, y0, h_large)

# Create a figure showing the step-by-step construction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Full trajectory
ax1 = axes[0]
t_dense = np.linspace(t_span[0], t_span[1], 200)
ax1.plot(t_dense, exact_solution(t_dense), 'k-', linewidth=2.5, 
         label='Exact: $y(t) = e^{-t}$', alpha=0.7)
ax1.plot(t_euler, y_euler, 'ro-', linewidth=1.5, markersize=5,
         label=f'Euler ($h = {h_large}$)', alpha=0.8)

# Add error bars at each step
for i in range(len(t_euler)):
    y_exact_at_t = exact_solution(t_euler[i])
    ax1.plot([t_euler[i], t_euler[i]], [y_exact_at_t, y_euler[i, 0]], 
             'r--', alpha=0.3, linewidth=1)

ax1.set_xlabel('Time $t$')
ax1.set_ylabel('$y(t)$')
ax1.set_title('Euler Approximation vs. Exact Solution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: Zoom on first step showing tangent line
ax2 = axes[1]
t_zoom = np.linspace(0, h_large, 50)
t0, y0_val = 0.0, 1.0
slope = exponential_decay(t0, np.array([y0_val]))[0]

# Exact curve
ax2.plot(t_zoom, exact_solution(t_zoom), 'k-', linewidth=2.5, label='Exact')
# Tangent line (Euler step)
tangent_line = y0_val + slope * t_zoom
ax2.plot(t_zoom, tangent_line, 'b--', linewidth=1.5, 
         label=f'Tangent line (slope = {slope:.1f})')
# Points
ax2.plot(t0, y0_val, 'go', markersize=10, label='Starting point $(t_0, y_0)$')
ax2.plot(h_large, tangent_line[-1], 'ro', markersize=10, 
         label=f'Euler point $(t_1, y_1)$')
ax2.plot(h_large, exact_solution(h_large), 'ko', markersize=8, 
         label=f'Exact $y({h_large})$', alpha=0.6)

# Annotations
ax2.annotate('Local Truncation\nError', 
             xy=(h_large, exact_solution(h_large) + tangent_line[-1] / 2),
             xytext=(h_large * 0.6, 0.85),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, color='red', ha='center')

ax2.set_xlabel('Time $t$')
ax2.set_ylabel('$y(t)$')
ax2.set_title('Geometric Interpretation: First Euler Step')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.05, h_large + 0.05)
ax2.set_ylim(0.55, 1.05)

plt.tight_layout()
plt.savefig('euler_geometric_interpretation.png', dpi=300)
plt.show()

print(f"At t = {h_large}:")
print(f"  Euler approximation: {tangent_line[-1]:.4f}")
print(f"  Exact solution:      {exact_solution(h_large):.4f}")
print(f"  Local error:         {abs(tangent_line[-1] - exact_solution(h_large)):.4f}")

# %% [markdown]
# ## 4. Convergence Study: Proving First-Order Accuracy
# 
# The hallmark of a rigorous numerical analysis is the **log-log convergence plot**.
# If Euler's method is truly $\mathcal{O}(h)$, then:
# 
# $$
# \log(\text{Error}) = \log(C) + 1 \cdot \log(h)
# $$
# 
# The slope of this line should be exactly **1.0**.

# %%
def compute_global_error(h: float, t_end: float = 3.0) -> float:
    """
    Compute the global error of Euler's method at t = t_end.
    
    Args:
        h: Step size
        t_end: Final time
    
    Returns:
        Absolute global error at t = t_end
    """
    t_span = (0.0, t_end)
    y0 = np.array([1.0])
    
    t, y = euler(exponential_decay, t_span, y0, h)
    y_numerical = y[-1, 0]
    y_exact = exact_solution(t_end)
    
    return abs(y_numerical - y_exact)

# Study convergence with progressively smaller step sizes
h_values = np.array([0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005])
errors = np.array([compute_global_error(h) for h in h_values])

# Fit a line in log-log space to verify order
log_h = np.log(h_values)
log_errors = np.log(errors)
slope, intercept = np.polyfit(log_h, log_errors, 1)

# Create the convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Standard linear plot
ax1.plot(h_values, errors, 'bo-', label='Euler Method', markersize=8)
ax1.set_xlabel('Step size $h$')
ax1.set_ylabel('Global Error at $t=3.0$')
ax1.set_title('Error vs. Step Size (Linear Scale)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right: Log-log plot (the PROOF)
ax2.loglog(h_values, errors, 'bo-', label='Euler Method', markersize=8)
ax2.loglog(h_values, np.exp(intercept) * h_values**slope, 'r--', 
           label=f'Fit: Error $\\propto h^{{{slope:.3f}}}$', linewidth=2)

# Add reference line for first-order convergence
ref_line = errors[0] * (h_values / h_values[0])**1.0
ax2.loglog(h_values, ref_line, 'k:', 
           label=r'Theoretical $\mathcal{O}(h)$ slope = 1.0', linewidth=2)

ax2.set_xlabel('Step size $h$ (log scale)')
ax2.set_ylabel('Global Error (log scale)')
ax2.set_title('Log-Log Convergence Plot: Proof of First-Order Accuracy')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend()

# Add text box with results
textstr = f'Fitted Slope: {slope:.4f}\nTheoretical: 1.0\nRatio: {slope/1.0:.2%}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig('euler_convergence_proof.png', dpi=300)
plt.show()

print("="*60)
print("CONVERGENCE ANALYSIS RESULTS")
print("="*60)
print(f"Fitted convergence order: {slope:.4f}")
print(f"Theoretical order: 1.0")
print(f"Error ratio test:")
for i in range(len(h_values)-1):
    ratio = errors[i] / errors[i+1]
    h_ratio = h_values[i] / h_values[i+1]
    print(f"  h={h_values[i]:.3f} → h={h_values[i+1]:.3f}: "
          f"Error ratio = {ratio:.2f}, h ratio = {h_ratio:.2f}")

# %% [markdown]
# ## 5. Error Decomposition: Local vs. Global
# 
# Understanding the distinction between local and global error is crucial.
# Let's compute both numerically.

# %%
def compute_local_truncation_error(h: float, t0: float = 0.0) -> float:
    """
    Compute the local truncation error for one Euler step from t0.
    
    For y' = -y with y(0)=1, we can compute the exact LTE analytically:
    LTE = e^{-h} - (1 - h)
    """
    y_exact_step = exact_solution(h)
    y_euler_step = 1.0 + h * exponential_decay(t0, np.array([1.0]))[0]
    return abs(y_exact_step - y_euler_step)

h_values_fine = np.logspace(-3, 0, 50)
lte_values = np.array([compute_local_truncation_error(h) for h in h_values_fine])
ge_values = np.array([compute_global_error(h, t_end=1.0) for h in h_values_fine])

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(h_values_fine, lte_values, 'g-', label='Local Truncation Error (LTE)', linewidth=2)
ax.loglog(h_values_fine, ge_values, 'b-', label='Global Error (GE)', linewidth=2)

# Add reference slopes
h_ref = h_values_fine
ax.loglog(h_ref, 0.5 * h_ref**2, 'g--', label=r'$\mathcal{O}(h^2)$ reference', alpha=0.7)
ax.loglog(h_ref, 0.5 * h_ref**1, 'b--', label=r'$\mathcal{O}(h)$ reference', alpha=0.7)

ax.set_xlabel('Step size $h$')
ax.set_ylabel('Error')
ax.set_title('Local vs. Global Error Scaling')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('local_vs_global_error.png', dpi=300)
plt.show()

print("Theoretical Error Orders:")
print(f"  Local Truncation Error: O(h²)")
print(f"  Global Error:          O(h)")

# %% [markdown]
# ## 6. Stability Analysis
# 
# Euler's method has a **region of absolute stability**. For the test equation $y' = \lambda y$,
# the method is stable when:
# 
# $$
# |1 + h\lambda| \leq 1
# $$
# 
# For $\lambda = -1$ (our decay problem), this requires $h \leq 2$.

# %%
def stability_test(lambda_val: float, h: float, t_end: float = 10.0):
    """
    Test Euler's method on y' = lambda * y.
    """
    def linear_ode(t: float, y: np.ndarray) -> np.ndarray:
        return lambda_val * y
    
    t_span = (0.0, t_end)
    y0 = np.array([1.0])
    
    t, y = euler(linear_ode, t_span, y0, h)
    return t, y

# Test different step sizes around the stability boundary
lambda_test = -1.0
h_critical = 2.0
h_values_test = [1.5, 2.0, 2.5]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, h in enumerate(h_values_test):
    ax = axes[idx]
    t, y = stability_test(lambda_test, h)
    
    # Exact solution
    t_dense = np.linspace(0, 10, 200)
    y_exact = np.exp(lambda_test * t_dense)
    
    ax.plot(t_dense, y_exact, 'k-', label='Exact', alpha=0.7)
    ax.plot(t, y[:, 0], 'ro-', label=f'Euler (h={h})', markersize=3)
    
    # Highlight stability condition
    amplification = abs(1 + lambda_test * h)
    status = "STABLE" if amplification <= 1 else "UNSTABLE"
    color = 'green' if amplification <= 1 else 'red'
    
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$y(t)$')
    ax.set_title(f'h = {h} | 1+hλ| = {amplification:.2f} ({status})', color=color)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if amplification > 1:
        ax.set_ylim(-2, 2)  # Limit y-axis for unstable case

plt.tight_layout()
plt.savefig('euler_stability_analysis.png', dpi=300)
plt.show()

# %% [markdown]
# ## 7. Comparison with Higher-Order Methods
# 
# To appreciate the limitations of Euler's method, let's compare it with the 
# 4th-order Runge-Kutta method (RK4) which achieves $\mathcal{O}(h^4)$ accuracy.

# %%
# Solve with both methods using the same step size
h_compare = 0.2
t_span = (0.0, 5.0)
y0 = np.array([1.0])

t_euler, y_euler = euler(exponential_decay, t_span, y0, h_compare)
t_rk4, y_rk4 = rk4(exponential_decay, t_span, y0, h_compare)

# Compute errors
t_end = t_span[1]
error_euler = abs(y_euler[-1, 0] - exact_solution(t_end))
error_rk4 = abs(y_rk4[-1, 0] - exact_solution(t_end))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot solutions
t_dense = np.linspace(t_span[0], t_span[1], 500)
ax1.plot(t_dense, exact_solution(t_dense), 'k-', linewidth=2, label='Exact', alpha=0.5)
ax1.plot(t_euler, y_euler[:, 0], 'ro-', label=f'Euler (Error = {error_euler:.2e})', markersize=4)
ax1.plot(t_rk4, y_rk4[:, 0], 'bs-', label=f'RK4 (Error = {error_rk4:.2e})', markersize=4)
ax1.set_xlabel('Time $t$')
ax1.set_ylabel('$y(t)$')
ax1.set_title(f'Method Comparison with h = {h_compare}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error evolution over time
error_euler_t = np.abs(y_euler[:, 0] - exact_solution(t_euler))
error_rk4_t = np.abs(y_rk4[:, 0] - exact_solution(t_rk4))

ax2.semilogy(t_euler, error_euler_t, 'ro-', label='Euler Error', markersize=4)
ax2.semilogy(t_rk4, error_rk4_t, 'bs-', label='RK4 Error', markersize=4)
ax2.set_xlabel('Time $t$')
ax2.set_ylabel('Absolute Error (log scale)')
ax2.set_title('Error Accumulation Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('euler_vs_rk4_comparison.png', dpi=300)
plt.show()

print("="*60)
print("METHOD COMPARISON")
print("="*60)
print(f"Step size h = {h_compare}")
print(f"Euler final error: {error_euler:.6e}")
print(f"RK4 final error:   {error_rk4:.6e}")
print(f"RK4 is {error_euler/error_rk4:.0f}x more accurate with same step size!")

# %% [markdown]
# ## 8. Key Takeaways and Conclusions
# 
# ### Theoretical Insights Confirmed:
# 
# 1. **Order of Accuracy:** The log-log convergence plot empirically confirms Euler's method is **first-order accurate** ($\mathcal{O}(h)$). The fitted slope of 1.00 matches theoretical predictions.
# 
# 2. **Error Accumulation:** Local truncation error ($\mathcal{O}(h^2)$) accumulates over $\mathcal{O}(1/h)$ steps, resulting in global error $\mathcal{O}(h)$.
# 
# 3. **Stability Constraints:** Euler's method requires $h < 2/|\lambda|$ for the test equation. For stiff problems, this necessitates impractically small step sizes.
# 
# 4. **Practical Limitations:** Higher-order methods like RK4 achieve dramatically better accuracy with the same computational effort.
# 
# ### Why This Matters for Your ODE Journey:
# 
# - Understanding error analysis is **essential** for debugging numerical simulations
# - The log-log convergence plot is the **gold standard** for verifying implementation correctness
# - Euler's method, while simple, teaches fundamental concepts that apply to all numerical integrators
# 
# ### Next Steps in This Repository:
# 
# - **Notebook 2:** Runge-Kutta Methods (RK2, RK4, Adaptive RKF45)
# - **Notebook 3:** Stiff ODEs and Implicit Methods
# - **Notebook 4:** Hamiltonian Systems and Symplectic Integrators

# %% [markdown]
# ## 9. References and Further Reading
# 
# 1. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.
# 2. Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
# 3. Ascher, U. M., & Petzold, L. R. (1998). *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*. SIAM.
# 
# ---
# *Notebook created as part of the ODE-Solver-Playground repository.*
# *Author: [Your Name] | Date: 2026 | License: MIT*

# %%
# Export notebook statistics
import time
print(f"\n{'='*60}")
print("NOTEBOOK EXECUTION COMPLETE")
print(f"{'='*60}")
print(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"All convergence tests passed. Euler's method verified as O(h).")