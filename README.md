# **From Points to Clouds: Learning Robust Semantic Distributions for Vision-Language Prompting**

> If you find this project useful, a star 🌟 would be greatly appreciated!

## 📌 Updates

* **2026.03** — Update ver. 
* **2025.11** — Initial project setup.

---

## 💡 Overview

While Multimodal Prompt Learning (MPL) effectively adapts Vision-Language Models, optimizing a single static point representation can be limited by base-class overfitting and training instability. Furthermore, addressing this via external Large Language Models (LLMs) often introduces additional computational overhead and reliance on external priors.

To overcome these limitations endogenously, we propose Points-to-Clouds (P2C), a novel LLM-free framework that reframes prompt learning as a dynamic denoising task. P2C transitions from learning a deterministic point to a continuous semantic cloud via a dual denoising mechanism.

<div align="center">
<img src="assets/Framework.png" width="850"/>
</div>

## 🧠 Key Contributions

### 🔹 1. Endogenous and LLM-Free

P2C fundamentally challenges the reliance on external explicit knowledge (e.g., querying LLMs for attribute generation). It learns a robust semantic region entirely from within, ensuring scalability and preventing hallucinations in specialized domains.

### 🔹 2. Dual Denoising Mechanism

We introduce a synergistic denoising approach. A Dynamic Prompt Denoising (DPD) module injects annealed GMM noise into text prompts. Simultaneously, an auxiliary consistency loss explicitly forces the V-L Mapper to reconstruct clean visual prompts from these perturbed inputs.

### 🔹 3. Fundamental Structural Stabilizer

By smoothing the optimization landscape, P2C acts as an effective stabilizer. It significantly mitigates the initialization variance commonly observed in prompt learning and enables reliable convergence regardless of prompt capacity or random seeds.

## 📊 Empirical Performance

Extensive experiments across 11 diverse datasets highlight the superiority of P2C. It achieves a state-of-the-art 79.7% harmonic mean on the challenging base-to-novel generalization benchmark, alongside excellent cross-dataset transferability.

### **Base-to-Novel Generalization (Average over 11 datasets)**

| Method         | Base     | Novel    | HM       |
| -------------- | -------- | -------- | -------- |
| CoOp           | 82.7     | 63.2     | 71.7     |
| CoCoOp         | 80.5     | 71.7     | 75.8     |
| MaPLe          | 82.3     | 75.1     | 78.6     |
| **P2C (Ours)** | **83.5** | **76.1** | **79.7** |

---

## 🛠 Installation

### Clone the repo

```bash
git clone https://github.com/vranlee/P2C.git
cd P2C
```

### Install dependencies

```bash
conda create -n p2c python=3.8 -y
conda activate p2c
pip install -r requirements.txt
```

---

## 🚀  Core Implementation Preview

We provide a sneak peek into the core implementation of our Dual Denoising Mechanism.

### **1. Dynamic Prompt Denoising (DPD)**

```python
# A sophisticated GMM noise generator for multi-modal semantic clouds
class GaussianMixtureNoiseGenerator(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.num_components = cfg.TRAINER.PROMPT_DENOISING.GMM_COMPONENTS
        self.mix_weights = torch.ones(self.num_components, device=device) / self.num_components
        self.means = torch.tensor(cfg.TRAINER.PROMPT_DENOISING.GMM_MEANS, device=device)
        self.stds = torch.tensor(cfg.TRAINER.PROMPT_DENOISING.GMM_STDS, device=device)

    def forward(self, tensor_like):
        mix = Categorical(self.mix_weights)
        comp = Normal(self.means, self.stds)
        gmm = MixtureSameFamily(mix, comp)
        return gmm.sample(tensor_like.shape).to(dtype=tensor_like.dtype)
```

### **2. Auxiliary V-L Mapper Denoising (The Stabilizer)**

```python
# Enforcing the V-L Mapper to act as a Denoising Autoencoder
if self.training and self.v_t_mapper_cfg.ENABLED:
    # 1. Clean visual target (sg operator applied via .detach())
    clean_text_prompt = self.prompt_learner.ctx
    clean_vision_prompt = shared_ctx.detach() 

    # 2. Independent noise sampling for the auxiliary task
    noise = torch.randn_like(clean_text_prompt) * self.aux_noise_std
    noisy_text_prompt = clean_text_prompt + noise

    # 3. Reconstruction and consistency loss computation
    reconstructed_vision_prompt = self.prompt_learner.proj(noisy_text_prompt)
    consistency_loss = F.mse_loss(reconstructed_vision_prompt, clean_vision_prompt)
    
    total_loss = classification_loss + self.aux_loss_weight * consistency_loss
```

---

## 📦 Project Structure

```
Scheduled to be publicly released upon paper acceptance.
```

---

## 📜 Citation

```
Scheduled to be released after the arXiv version.
```