
# 🧠 Custom Transformer Variants for Machine Translation

This repository contains three versions of a custom Transformer architecture built from scratch for machine translation tasks. Each version explores different design choices to understand their impact on performance and interpretability.

## 📦 Variants

1. **Dense Transformer**  
   - Standard Transformer with full dense layers.
2. **Transformer with RoPE**  
   - Integrates Rotary Positional Embedding (RoPE) into the attention mechanism.
3. **Transformer with MoE**  
   - Adds Mixture-of-Experts (MoE) layers for conditional computation and scalability.
   - ✅ Supports **multiple MoE routing strategies** in encoder and decoder:
     - MoE in **odd** layers
     - MoE in **even** layers
     - MoE in **first half** of the layers
     - MoE in **second half** of the layers  
   - You can select one of these variants by modifying the final argument in the `build_encoder_layers(...)` and `build_decoder_layers(...)` functions inside `model_architecture.py` (typically a lambda like `moe_odd`, `moe_even`, etc.).


Each version is organized in its own folder with dedicated training and evaluation scripts.

---

## 🛠️ Usage

### 🔧 Training

To train a model, run:

```bash
python main.py
```

from within the respective version folder:
- `DenseTransformer/`
- `TransformerWithRoPE/`
- `TransformerWithMoE/`

Default is 5 epochs but it can changed in the model_train function
---

### 📏 Evaluation (Metrics)

To compute translation metrics such as BLEU or accuracy, use:

```bash
python compute_metrics.py
```

inside the appropriate version folder.

---

### 🎯 Attention Visualization

To visualize attention heads and patterns, run:

```bash
python visualise_attention.py
```

inside the corresponding variant’s folder.

---

## 📁 Folder Structure

```
Transformer_MT/
├── DenseTransformer/
│   ├── main.py
│   ├── compute_metrics.py
│   └── visualise_attention.py
├── TransformerWithRoPE/
│   ├── main.py
│   ├── compute_metrics.py
│   └── visualise_attention.py
├── TransformerWithMoE/
│   ├── main.py
│   ├── compute_metrics.py
│   └── visualise_attention.py
└── README.md
```

---

## 📜 License

MIT License


---

## ⚙️ Configuration

Before running any script, make sure to adjust the following settings inside each version's directory:

- **Checkpoint Selection**:  
  Specify the path to the model checkpoint you want to use during evaluation or visualization in `compute_metrics.py` and `visualise_attention.py` in load_checkpoint function.

- **Attention Visualization**:  
  Choose which **layer** and **head** to visualize by editing the corresponding parameters inside `visualise_attention.py`.

These settings ensure you get the correct outputs for your specific experiment or use case.
