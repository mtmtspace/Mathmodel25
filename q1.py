import numpy as np
import matplotlib.pyplot as plt

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_sprt_simulation(p0, p1, alpha, beta, true_p, max_steps=1000):
    """
    Simulates a single run of the Sequential Probability Ratio Test (SPRT).

    Args:
        p0 (float): Defect rate under the null hypothesis (H0).
        p1 (float): Defect rate under the alternative hypothesis (H1).
        alpha (float): Type I error probability.
        beta (float): Type II error probability.
        true_p (float): The actual defect rate of the batch.
        max_steps (int): Maximum number of samples to draw.

    Returns:
        tuple: A tuple containing the list of LLR values and the final decision.
    """
    # Calculate acceptance and rejection boundaries
    A = np.log(beta / (1 - alpha))
    B = np.log((1 - beta) / alpha)

    # Log-likelihood ratio increments
    log_ratio_success = np.log((1 - p1) / (1 - p0)) # for a good item
    log_ratio_failure = np.log(p1 / p0)           # for a defective item

    llr_path = [0]
    current_llr = 0

    for i in range(1, max_steps + 1):
        # Simulate sampling one item from the batch
        is_defective = np.random.rand() < true_p
        
        if is_defective:
            current_llr += log_ratio_failure
        else:
            current_llr += log_ratio_success
        
        llr_path.append(current_llr)

        # Check against boundaries
        if current_llr >= B:
            return llr_path, f"拒绝 (样本数: {i})"
        if current_llr <= A:
            return llr_path, f"接受 (样本数: {i})"

    return llr_path, f"达到最大样本数仍无法决策"

# --- Parameters based on the problem ---
p0 = 0.10      # Nominal value (H0)
p1 = 0.15      # Unacceptable defect rate (H1) - a reasonable choice
alpha = 0.05   # 1 - 95% confidence for rejection
beta = 0.10    # 1 - 90% confidence for acceptance

# Calculate boundaries for plotting
A_boundary = np.log(beta / (1 - alpha))
B_boundary = np.log((1 - beta) / alpha)

# --- Run simulations and plot ---
plt.figure(figsize=(14, 8))

# Scenarios to simulate
scenarios = {
    "真实次品率 5% (合格批次)": 0.05,
    "真实次品率 10% (边界批次)": 0.10,
    "真实次品率 15% (边界批次)": 0.15,
    "真实次品率 20% (不合格批次)": 0.20
}

# Run multiple paths for each scenario for better visualization
for name, p in scenarios.items():
    for _ in range(3): # Plot 3 sample paths for each scenario
        path, decision = run_sprt_simulation(p0, p1, alpha, beta, p)
        plt.plot(path, marker='o', markersize=3, linestyle='-', label=f"{name} - {decision}")

# Plot boundaries
plt.axhline(y=A_boundary, color='green', linestyle='--', linewidth=2, label=f'接受边界 A ≈ {A_boundary:.2f}')
plt.axhline(y=B_boundary, color='red', linestyle='--', linewidth=2, label=f'拒绝边界 B ≈ {B_boundary:.2f}')

# Formatting the plot
plt.title('问题一：SPRT抽样检测过程模拟', fontsize=16)
plt.xlabel('抽样数量 (件)', fontsize=12)
plt.ylabel('对数似然比 (LLR)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('q1_sprt.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Provide specific results for the two required scenarios ---
print("\n--- 针对问题要求的具体方案 ---")
print(f"参数设定: p0={p0}, p1={p1}, alpha={alpha}, beta={beta}")
print(f"计算得出的接受边界 A = {A_boundary:.4f}")
print(f"计算得出的拒绝边界 B = {B_boundary:.4f}")
print("\n决策流程:")
print("1. 从批次中逐一抽取零配件进行检测。")
print("2. 每检测一个，根据其是否为次品，更新对数似然比(LLR)值。")
print(f"   - 若为合格品, LLR = LLR + {np.log((1-p1)/(1-p0)):.4f}")
print(f"   - 若为次品,   LLR = LLR + {np.log(p1/p0):.4f}")
print(f"3. 判断更新后的LLR值:")
print(f"   - 如果 LLR <= {A_boundary:.4f}, 立即停止检测，做出“接收”决策。")
print(f"   - 如果 LLR >= {B_boundary:.4f}, 立即停止检测，做出“拒收”决策。")
print("   - 如果 LLR 介于两者之间，则继续抽样检测下一个零配件。")
