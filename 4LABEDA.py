import pandas as pd
import numpy as np
import scipy.stats as st
import math
import matplotlib.pyplot as plt

# =========================
# "display" compatible con .py y Jupyter
# =========================
try:
    from IPython.display import display
except ImportError:
    def display(x):
        if hasattr(x, "to_string"):
            print(x.to_string(index=False))
        else:
            print(x)

# =========================
# Helpers
# =========================
def cohen_d(x, y):
    """Cohen's d for independent samples (pooled SD)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / sp

def welch_ttest(x, y, alternative="two-sided"):
    """Welch's t-test (independent samples, unequal variances)."""
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values

    t, p_two = st.ttest_ind(x, y, equal_var=False)

    if alternative == "two-sided":
        return t, p_two

    if alternative == "greater":  # H1: mean(x) > mean(y)
        p = p_two / 2 if t > 0 else 1 - p_two / 2
        return t, p

    if alternative == "less":  # H1: mean(x) < mean(y)
        p = p_two / 2 if t < 0 else 1 - p_two / 2
        return t, p

    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons."""
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # Holm adjusted p-values
    adj_sorted = np.empty(m)
    for i, p in enumerate(sorted_p):
        adj_sorted[i] = min(1.0, (m - i) * p)

    # enforce monotonicity
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

    reject_sorted = adj_sorted <= alpha

    # back to original order
    adj = np.empty(m)
    reject = np.empty(m, dtype=bool)
    adj[order] = adj_sorted
    reject[order] = reject_sorted

    return reject, adj

# ==========================================================
# CHALLENGE 1 — Pokemon (Hypothesis Testing)
# ==========================================================
pokemon_url = "https://raw.githubusercontent.com/data-bootcamp-v4/data/main/pokemon.csv"
df_poke = pd.read_csv(pokemon_url)

print("Pokemon shape:", df_poke.shape)
print("Pokemon columns:\n", df_poke.columns)

# -------------------------
# 1.1 Dragon has higher HP than Grass (one-sided Welch t-test)
# H0: mean_HP(Dragon) <= mean_HP(Grass)
# H1: mean_HP(Dragon)  > mean_HP(Grass)
# -------------------------
dragon_hp = df_poke.loc[(df_poke["Type 1"] == "Dragon") | (df_poke["Type 2"] == "Dragon"), "HP"]
grass_hp  = df_poke.loc[(df_poke["Type 1"] == "Grass")  | (df_poke["Type 2"] == "Grass"),  "HP"]

t_stat, p_val = welch_ttest(dragon_hp, grass_hp, alternative="greater")
d = cohen_d(dragon_hp, grass_hp)

print("\n--- Challenge 1.1: Dragon HP > Grass HP ---")
print("n_dragon:", dragon_hp.dropna().shape[0], "| mean:", dragon_hp.mean())
print("n_grass :", grass_hp.dropna().shape[0],  "| mean:", grass_hp.mean())
print("Welch t-stat:", t_stat)
print("One-sided p-value:", p_val)
print("Cohen's d (Dragon - Grass):", d)

alpha = 0.05
if p_val < alpha:
    print("✅ Reject H0 at 5%: Dragons have significantly higher HP on average than Grass.")
else:
    print("❌ Fail to reject H0 at 5%: Not enough evidence that Dragons have higher HP than Grass.")

# -------------------------
# 1.2 Legendary vs Non-Legendary have different stats (multiple Welch t-tests + Holm)
# -------------------------
stats_cols = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

leg = df_poke[df_poke["Legendary"] == True]
non = df_poke[df_poke["Legendary"] == False]

print("\n--- Challenge 1.2: Legendary vs Non-Legendary stats ---")
print("n_legendary:", leg.shape[0], "| n_nonlegendary:", non.shape[0])

results = []
pvals = []

for col in stats_cols:
    x = leg[col]
    y = non[col]
    t, p = welch_ttest(x, y, alternative="two-sided")
    dcol = cohen_d(x, y)
    results.append((col, x.mean(), y.mean(), t, p, dcol))
    pvals.append(p)

reject, p_adj = holm_bonferroni(pvals, alpha=0.05)

out = pd.DataFrame(
    results,
    columns=["stat", "mean_legendary", "mean_nonlegendary", "t_stat", "p_value", "cohen_d"]
)
out["p_holm"] = p_adj
out["reject_0.05"] = reject
out = out.sort_values("p_holm")

print("\nResults table (Holm-corrected):")
display(out)

sig_stats = out[out["reject_0.05"] == True]["stat"].tolist()
if sig_stats:
    print("\n✅ Significant differences (Holm corrected, 5%) in:", sig_stats)
else:
    print("\n❌ No significant differences after Holm correction at 5%.")

# ==========================================================
# CHALLENGE 2 — California Housing
# ==========================================================
housing_url = "https://raw.githubusercontent.com/data-bootcamp-v4/data/main/california_housing.csv"
df_house = pd.read_csv(housing_url)

print("\nCalifornia housing shape:", df_house.shape)
print("California housing columns:\n", df_house.columns)

# Hypothesis:
# Houses close to either a school or a hospital are more expensive.
# School coords: (-118, 34)
# Hospital coords: (-122, 37)
# Close if min(distance_to_school, distance_to_hospital) < 0.50
# Test: independent Welch t-test, one-sided (close > far)

school = (-118, 34)
hospital = (-122, 37)

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

df_house = df_house.copy()
df_house["dist_school"] = euclid_dist(df_house["longitude"], df_house["latitude"], school[0], school[1])
df_house["dist_hospital"] = euclid_dist(df_house["longitude"], df_house["latitude"], hospital[0], hospital[1])
df_house["dist_min"] = df_house[["dist_school", "dist_hospital"]].min(axis=1)

threshold = 0.50
df_house["is_close"] = df_house["dist_min"] < threshold

close_vals = df_house.loc[df_house["is_close"] == True, "median_house_value"]
far_vals   = df_house.loc[df_house["is_close"] == False, "median_house_value"]

t2, p2 = welch_ttest(close_vals, far_vals, alternative="greater")
d2 = cohen_d(close_vals, far_vals)

print("\n--- Challenge 2: Close to school/hospital => more expensive ---")
print("Threshold distance:", threshold)
print("n_close:", close_vals.dropna().shape[0], "| mean:", close_vals.mean(), "| median:", close_vals.median())
print("n_far  :", far_vals.dropna().shape[0],   "| mean:", far_vals.mean(),   "| median:", far_vals.median())
print("Welch t-stat:", t2)
print("One-sided p-value:", p2)
print("Cohen's d (Close - Far):", d2)

if p2 < 0.05:
    print("✅ Reject H0 at 5%: Homes close to a school or hospital are significantly more expensive on average.")
else:
    print("❌ Fail to reject H0 at 5%: Not enough evidence that close homes are more expensive.")

# Optional: quick visualization
plt.figure(figsize=(6, 4))
plt.boxplot([close_vals.dropna(), far_vals.dropna()], labels=["Close", "Far"], showfliers=False)
plt.title("Median house value: Close vs Far")
plt.ylabel("median_house_value")
plt.tight_layout()
plt.show()
