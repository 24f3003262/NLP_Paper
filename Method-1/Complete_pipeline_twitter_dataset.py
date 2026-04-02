import jax
from jax import jit
import jax.numpy as jnp
import optax
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import copy
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve
)

# --- 1. CONFIGURATION ---
BATCH_SIZE = 512
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_CLASSES = 3

# --- 2. DATA LOAD ---
print("--- Loading Twitter_Data.csv ---")
df = pd.read_csv('Twitter_Data.csv').dropna(subset=['clean_text', 'category'])
label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
df['category'] = df['category'].map(label_map).astype(int)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_in_chunks(texts, batch_size=5000):
    all_ids = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size].tolist()
        tokens = tokenizer(chunk, truncation=True, padding="max_length", max_length=64, return_tensors="np")["input_ids"]
        all_ids.append(tokens)
    return jnp.concatenate([jnp.array(x) for x in all_ids], axis=0)

print("--- Tokenizing Dataset ---")
train_ids = jax.device_put(tokenize_in_chunks(train_df["clean_text"]))
train_labels = jax.device_put(jnp.array(train_df["category"].values, dtype=jnp.int32))
test_ids = jax.device_put(tokenize_in_chunks(test_df["clean_text"]))
test_labels = jax.device_put(jnp.array(test_df["category"].values, dtype=jnp.int32))

# --- 3. MODEL FUNCTIONS ---
def init_params_research(vocab_size):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    params = {
        'w': jax.random.normal(k1, (vocab_size,)) * 0.01,
        'emb': jax.random.normal(k2, (vocab_size, EMBED_DIM)) * 0.01,
        'W1': jax.nn.initializers.glorot_normal()(k3, (EMBED_DIM, HIDDEN_DIM)),
        'b1': jnp.zeros(HIDDEN_DIM),
        'W2': jax.nn.initializers.glorot_normal()(k4, (HIDDEN_DIM, 128)),
        'b2': jnp.zeros(128),
        'W3': jax.nn.initializers.glorot_normal()(k5, (128, NUM_CLASSES)),
        'b3': jnp.zeros(NUM_CLASSES)
    }
    return jax.device_put(params)

@jax.jit
def total_loss_fn(params, token_ids, labels, l1_val=1.0, l2_val=0.01, temp=1.0):
    w_sig = jax.nn.sigmoid(params['w'] * temp)
    batch_w = w_sig[token_ids]
    X_prime = jnp.mean(params['emb'][token_ids] * batch_w[:, :, jnp.newaxis], axis=1)

    diff = X_prime[:, jnp.newaxis, :] - X_prime[jnp.newaxis, :, :]
    R = jnp.exp(-jnp.sum(diff**2, axis=-1) / 2.0)
    enemy_mask = labels[:, jnp.newaxis] != labels[jnp.newaxis, :]
    mu = 1.0 - (jax.nn.logsumexp(10.0 * jnp.where(enemy_mask, R, -1e9), axis=1) / 10.0)
    l_rs = 1.0 - jnp.mean(mu)

    h1 = jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1'])
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
    logits = jnp.dot(h2, params['W3']) + params['b3']

    l_ce = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, NUM_CLASSES)))
    l_sparsity = jnp.mean(w_sig)
    return l_ce + (l1_val * l_rs) + (l2_val * l_sparsity)

@jax.jit
def evaluate_metrics_gpu(params, x_ids, y, temp=1.0):
    w_sig = jax.nn.sigmoid(params['w'] * temp)
    X_prime = jnp.mean(params['emb'][x_ids] * w_sig[x_ids][:, :, jnp.newaxis], axis=1)
    h1 = jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1'])
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
    logits = jnp.dot(h2, params['W3']) + params['b3']
    probs = jax.nn.softmax(logits)
    preds = jnp.argmax(logits, axis=1)
    acc = jnp.mean(preds == y)
    ll = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, NUM_CLASSES)))
    k = jnp.sum(w_sig > 0.5)
    return preds, probs, acc, ll, k

@jax.jit
def evaluate_baseline_direct(params, x_ids, y):
    X_prime = jnp.mean(params['emb'][x_ids], axis=1)
    h1 = jax.nn.relu(jnp.dot(X_prime, params['W1']) + params['b1'])
    h2 = jax.nn.relu(jnp.dot(h1, params['W2']) + params['b2'])
    logits = jnp.dot(h2, params['W3']) + params['b3']
    probs = jax.nn.softmax(logits)
    preds = jnp.argmax(logits, axis=1)
    ll = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, NUM_CLASSES)))
    k_baseline = params['emb'].shape[0]
    return preds, probs, ll, k_baseline

# --- HELPER: ADVANCED METRICS ---
def calculate_advanced_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    recalls = np.diag(cm) / (np.sum(cm, axis=1) + 1e-9)
    g_mean = np.exp(np.mean(np.log(recalls + 1e-9)))
    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    auc_roc = roc_auc_score(y_true_oh, y_probs, multi_class='ovr', average='weighted')
    auprc = average_precision_score(y_true_oh, y_probs, average='weighted')
    return acc, f1, prec, rec, mcc, kappa, g_mean, auc_roc, auprc, cm

def plot_research_curves(y_true, y_probs, title_prefix, filename):
    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {i}')
    plt.plot([0, 1], [0, 1], 'k--'); plt.title(f'{title_prefix} ROC'); plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.title(f'{title_prefix} PR Curve'); plt.legend()
    plt.savefig(filename); plt.show()

# --- 4. EXECUTION LOOP ---
optimizer = optax.adam(learning_rate=LEARNING_RATE)
best_aic, best_params, best_epoch = float('inf'), None, -1
results_history, prev_feature_set = [], set()
params = init_params_research(tokenizer.vocab_size)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, b_x, b_y, l1, l2, temp):
    loss_val, grads = jax.value_and_grad(total_loss_fn)(params, b_x, b_y, l1, l2, temp)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), next_opt_state, loss_val

print(f"--- Starting {NUM_EPOCHS} Iteration Run ---")

for epoch in range(NUM_EPOCHS):
    idx = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_ids))
    epoch_x, epoch_y = train_ids[idx], train_labels[idx]
    epoch_losses = []
    for i in range(0, len(epoch_x), BATCH_SIZE):
        b_x, b_y = epoch_x[i:i+BATCH_SIZE], epoch_y[i:i+BATCH_SIZE]
        if len(b_x) < BATCH_SIZE: continue
        params, opt_state, loss = train_step(params, opt_state, b_x, b_y, 1.0, 0.01, 1.0)
        epoch_losses.append(loss)

    # RESTORED: Unpacking matches the 5 return values of evaluate_metrics_gpu
    preds, probs, acc, ll, k = evaluate_metrics_gpu(params, test_ids, test_labels)
    y_true, y_pred, y_probs = jax.device_get(test_labels), jax.device_get(preds), jax.device_get(probs)
    
    acc_val, f1_val, prec_val, rec_val, mcc_val, kappa_val, gmean_val, auc_val, auprc_val, _ = calculate_advanced_metrics(y_true, y_pred, y_probs)

    roughness = 1.0 - float(jnp.mean(jax.nn.sigmoid(1.0 - ll)))
    w_sig = jax.device_get(jax.nn.sigmoid(params['w']))
    current_feature_set = set(np.where(w_sig > 0.5)[0])
    jaccard = len(current_feature_set & prev_feature_set) / len(current_feature_set | prev_feature_set) if prev_feature_set else 0.0
    prev_feature_set = current_feature_set

    n = test_ids.shape[0]
    cur_aic = float(2 * int(k) + 2 * ll * n)
    cur_bic = float(int(k) * jnp.log(n) + 2 * ll * n)
    reduct_pct = (1 - (int(k) / tokenizer.vocab_size)) * 100

    if cur_aic < best_aic:
        best_aic, best_epoch = cur_aic, epoch
        best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
        print(f"[*] New Best Model | Epoch {epoch} | AIC: {cur_aic:.2f} | k: {int(k)}")

    results_history.append([
        epoch, f"{np.mean(epoch_losses):.4f}", acc_val, prec_val, rec_val, f1_val, mcc_val, kappa_val, gmean_val, auc_val, auprc_val,
        float(ll), cur_aic, cur_bic, int(k), reduct_pct, roughness, jaccard
    ])
    print(f"Epoch {epoch:02d} | F1: {f1_val:.4f} | BIC: {cur_bic:.2f} | k: {int(k)} | Stab: {jaccard:.4f}")

# --- 5. EXPORTS ---
print("--- Exporting Forensic Reports ---")

# 1. Best Reduced Model Final Stats
p_r, pr_r, _, ll_r, k_r = evaluate_metrics_gpu(best_params, test_ids, test_labels)
acc_r, f1_r, pr_r_val, re_r_val, mcc_r, kappa_r, g_r, auc_r, prc_r, cm_r = calculate_advanced_metrics(y_true, jax.device_get(p_r), jax.device_get(pr_r))
pd.DataFrame([{
    "k": int(k_r), "Accuracy": acc_r, "Precision": pr_r_val, "Recall": re_r_val, "F1": f1_r, 
    "MCC": mcc_r, "Kappa": kappa_r, "G-Mean": g_r, "AUC": auc_r, "AUPRC": prc_r, "AIC": best_aic, "Roughness": 1.0 - float(jnp.mean(jax.nn.sigmoid(1.0 - ll_r)))
}]).to_csv("reduced_final_metrics.csv", index=False)
plot_research_curves(y_true, jax.device_get(pr_r), "Reduced DRSAR-Net", "reduced_curves.png")

# 2. Original Baseline Final Stats
p_b, pr_b, ll_b, k_b = evaluate_baseline_direct(best_params, test_ids, test_labels)
acc_b, f1_b, pr_b_val, re_b_val, mcc_b, kappa_b, g_b, auc_b, prc_b, _ = calculate_advanced_metrics(y_true, jax.device_get(p_b), jax.device_get(pr_b))
pd.DataFrame([{
    "k": int(k_b), "Accuracy": acc_b, "Precision": pr_b_val, "Recall": re_b_val, "F1": f1_b, 
    "MCC": mcc_b, "Kappa": kappa_b, "G-Mean": g_b, "AUC": auc_b, "AUPRC": prc_b, "AIC": float(2*k_b + 2*ll_b*n)
}]).to_csv("baseline_final_metrics.csv", index=False)
plot_research_curves(y_true, jax.device_get(pr_b), "Original Baseline", "baseline_curves.png")

# 3. Training History CSV
h_cols = ["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1", "MCC", "Kappa", "G-Mean", "AUC", "AUPRC", "LogLoss", "AIC", "BIC", "k", "Reduct%", "Roughness", "Jaccard_Index"]
pd.DataFrame(results_history, columns=h_cols).to_csv("training_history_all_epochs.csv", index=False)

# 4. BEST REDUCT FEATURES EXPORT
w_final = jax.nn.sigmoid(best_params['w'])
indices = np.where(w_final > 0.5)[0]
id_to_word = {v: k for k, v in tokenizer.get_vocab().items()}
reduced_words = [{"word": id_to_word[int(idx)], "weight": float(w_final[idx])} for idx in indices]
pd.DataFrame(reduced_words).sort_values(by="weight", ascending=False).to_csv("best_reduct_features.csv", index=False)

print("Done. Separate CSVs, Reduced Features, and Curves generated.")