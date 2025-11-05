# 🧠 On Convergence of Adam with Data-Dependent Stepsize

**Authors:**  
Alokendu Mazumder¹, Rishabh Sabharwal², Manan Tayal¹, Bhartendu Kumar³, Arnab Roy¹, Chirag Garg⁴, Punit Rathore¹  

**Affiliations:**  
¹ Robert Bosch Center for Cyber Physical Systems, Indian Institute of Science (IISc) Bengaluru. 
² The University of Edinburgh.
³ Microsoft Research India. 
⁴ Department of Computer Science and Engineering, National Institute of Technology (NIT), Raipur.


---

## 📄 Overview

This repository contains the implementation and experiments for the paper **"On Convergence of Adam with Data-Dependent Stepsize"**.

We propose a **data- and network-driven constant stepsize** for the Adam optimizer that depends on the initial loss, an estimated Lipschitz smoothness, and the training horizon.  
This approach avoids tedious hyperparameter tuning and ensures stable convergence without learning rate decay.

The main update rule is expressed as:

```
αₒᵤᵣₛ = √( 2·L(w₀) / ( K̂·T ) )
```

Where:  
- **L(w₀)** → Initial loss  
- **K̂** → Estimated Lipschitz smoothness  
- **T** → Training horizon (number of iterations/epochs)

---

## 🚀 Key Contributions

- ✅ Introduces a **constant stepsize** formulation based on theoretical convergence analysis.  
- ✅ Provides **global convergence guarantees** for Adam using the above rule.  
- ✅ Demonstrates stability and strong empirical performance across CNNs, Transformers, and LLMs.  
- ✅ Includes reproducible experiments with PyTorch implementations.

---

## 🧩 Repository Structure

```
├── main.py                # Main entry point for running experiments
├── models.py              # Model architectures (CNNs, MLPs, ViT)
├── get_data.py            # Dataset loading utilities (MNIST, CIFAR, ImageNet)
├── get_scheduler.py       # Scheduler definitions for baseline comparisons
├── experiments.py         # Experimental setup and evaluation scripts
├── utils.py               # Helper functions, metrics, and logging
└── ieee_tai_adam_revised_v2_track.pdf  # Paper reference
```

---

## ⚙️ Installation

### Requirements
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- NumPy, Matplotlib, tqdm, torchvision  

### Setup
```bash
git clone https://github.com/<your-username>/adam-data-dependent-stepsize.git
cd adam-data-dependent-stepsize
pip install -r requirements.txt
```

---

## 💡 Usage

### Train with Proposed Stepsize
```bash
python main.py --model resnet18 --dataset cifar10 --optimizer adam --stepsize ours
```

### Train with Baseline Scheduler
```bash
python main.py --model resnet18 --dataset cifar10 --scheduler cosine --lr 0.001
```

### Estimate Lipschitz Constant
```bash
python experiments.py --estimate-lipschitz --model resnet18 --dataset cifar10
```

---

## 📊 Results Summary

### **Table I** — Top-1 Accuracy (%) with Different Learning Rate Schedulers and ADAM (ResNet-18, ImageNet)

| Scheduler | Step | Linear | Cosine | Exp. | Inv. Time | Sqrt | Ours (5.9×10⁻⁴) | 2×Ours | Ours/2 |
|------------|------|---------|---------|------|------------|-------|----------------|---------|--------|
| **Top-1 Acc. (%)** | 57.33 | 62.77 | 62.53 | 62.59 | 60.59 | 62.28 | **67.53** | *66.71* | **67.79** |

---

### **Table II** — Top-1 Accuracy (%) with Various Optimizers (ResNet-18, ImageNet)

| Optimizer | AdaBelief | AdaBound† | Yogi† | Adam† | MSVAG | RAdam‡ | AdamW† | Adam + Ours (5.9×10⁻⁴) | Adam + 2×Ours | Adam + Ours/2 |
|------------|------------|------------|--------|--------|--------|---------|---------|--------------------------|----------------|----------------|
| **Top-1 Acc. (%)** | **70.08** | 68.13 | 68.23 | 63.79 | 65.99 | 67.62 | 67.93 | **67.53** | *66.71* | **67.79** |

---

### **Table III** — LLM Generative Experiment. Dataset: SQuAD, Optimizer: Adam, Task: QA (generation)

| Metric | Model | Step | Linear | Cosine | Exp. | Sqrt | Inv-Time | Ours | 2×Ours | Ours/2 |
|---------|--------|------|---------|---------|-------|-------|-----------|-------|--------|--------|
| **Exact Match** | LLaMA-3-2.3B-Instruct | 0.7061 | 0.7034 | 0.7096 | 0.7013 | 0.7032 | 0.7046 | 0.6998 | 0.6941 | **0.7141** |
| | Qwen-3-4B-Instruct | 0.7173 | 0.7153 | 0.7189 | 0.7108 | 0.7123 | 0.7131 | 0.7135 | 0.6965 | **0.7176** |
| **F1** | LLaMA-3-2.3B-Instruct | 0.8546 | 0.8531 | 0.8572 | 0.8518 | 0.8523 | 0.8529 | 0.8508 | 0.8408 | **0.8585** |
| | Qwen-3-4B-Instruct | 0.8661 | 0.8642 | 0.8653 | 0.8621 | 0.8630 | 0.8629 | 0.8565 | 0.8577 | **0.8660** |

---

### **Table IV** — LLM Generative Experiment. Dataset: SQuAD, Scheduler: Cosine, Task: QA (generation)

| Metric | Model | AdaBelief | AdaBound | AdamW | Yogi | RAdam | Adam + Ours | Adam + 2×Ours | Adam + Ours/2 |
|---------|--------|------------|------------|--------|--------|--------|--------------|----------------|----------------|
| **Exact Match** | LLaMA-3-2.3B-Instruct | 0.7122 | 0.6836 | 0.7136 | 0.7088 | **0.7202** | 0.6985 | 0.6941 | **0.7141** |
| | Qwen-3-4B-Instruct | 0.7275 | 0.6992 | 0.7333 | 0.7170 | **0.7322** | 0.7135 | 0.6965 | **0.7176** |
| **F1** | LLaMA-3-2.3B-Instruct | 0.8605 | 0.8345 | 0.8636 | 0.8529 | **0.8648** | 0.8508 | 0.8408 | **0.8585** |
| | Qwen-3-4B-Instruct | 0.8701 | 0.8487 | 0.8728 | 0.8612 | **0.8724** | 0.8575 | 0.8477 | **0.8660** |

---

## 🧠 Theoretical Highlights

We prove that under mild smoothness assumptions:  
- Both deterministic and stochastic Adam converge to critical points.  
- Gradient norms vanish at a rate proportional to **O(T⁻¹ᐟ⁴)**.  
- The constant stepsize derived from data and model dynamics guarantees stability.

---

## 🧩 Citation

If you use this work, please cite:
```bibtex
@article{mazumder2023theoretical,
  title={A Theoretical and Empirical Study on the Convergence of Adam with an" Exact" Constant Step Size in Non-Convex Settings},
  author={Mazumder, Alokendu and Sabharwal, Rishabh and Tayal, Manan and Kumar, Bhartendu and Rathore, Punit},
  journal={arXiv preprint arXiv:2309.08339},
  year={2023}
}
```

---

## 🤝 Acknowledgements
This research was supported by [The Prime Minister's Research Fellowship].
