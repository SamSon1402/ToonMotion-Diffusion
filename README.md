# 🎭 ToonMotion-Diffusion

### *Teaching machines to animate cartoon characters — one text prompt at a time.*

---

<img width="1165" height="883" alt="image" src="https://github.com/user-attachments/assets/67e942d3-9137-43e8-82aa-4513a41b302c" />


Back in the golden age of animation, every single frame was drawn by hand. A team of artists would spend weeks bringing a character to life — sketching key poses, filling in the in-between frames, checking the timing, doing it all over again. Beautiful work. Painstaking work.

Then computers came along and made things faster, but the core problem stayed the same: **animating a character is slow, expensive, and repetitive.** A single second of 3D animation can take hours to produce. Most of that time isn't creative — it's mechanical.

This project asks a simple question: **What if you could just describe what a character should do, and the machine figures out the motion?**

Not for realistic humans walking through a park. For **cartoon characters** — the ones with giant heads, tiny bodies, duck feet, and elephant ears. The ones that move in snappy, exaggerated, physically impossible ways that make kids laugh.

That's what ToonMotion-Diffusion does.

---

## What It Does

You type a sentence like:

> *"Pocoyo jumps excitedly and waves both arms"*

<img width="1165" height="876" alt="image" src="https://github.com/user-attachments/assets/90622235-db85-4a92-a6d1-61aa0187e971" />


The model generates a complete 3D motion sequence — 120 frames at 24fps — with the exact joint rotations needed to animate that character in Maya or any 3D software.

The trick is that this works on **non-human cartoon rigs**. Every existing text-to-motion model in the research literature is trained on human motion capture data. Humans have predictable proportions — head is 1/7 of body height, arms reach to mid-thigh, knees bend one way. Cartoon characters break all of those rules.

Pocoyo's head is almost as big as his body. Pato is literally a duck. Maya the Bee has wings instead of arms. You can't just fine-tune a human motion model and expect it to work.

So we built a **Toon-Adapter** — a module that learns the specific skeleton, joint limits, and movement style of each character. The diffusion model generates motion in a universal space, and the adapter translates it into character-specific rig controller values.

---

## How It Works

The pipeline has four stages, like a classic animation workflow:

**Stage 1 — Understanding the Script** *(Text Encoding)*

The text prompt goes through a CLIP encoder, which converts words into a dense mathematical representation. The model doesn't read English — it reads a 512-dimensional vector that captures the meaning of "jumps excitedly and waves."

**Stage 2 — Starting from Noise** *(Forward Diffusion)*

We begin with pure random noise — imagine TV static, but in the shape of a motion sequence. This is 120 frames of complete randomness. The magic of diffusion models is that they learn to gradually remove this noise, step by step, until clean motion emerges.

**Stage 3 — Sculpting the Motion** *(Reverse Diffusion with Transformer)*

Over 50 denoising steps, a Transformer neural network looks at the noisy motion, the text description, and the character identity, then predicts what noise to remove. Early steps establish the broad structure — is the character jumping or walking? Later steps refine the details — how high does the arm wave, how snappy is the landing?

The Transformer uses:
- **Self-attention** over time — so frame 30 knows what happened at frame 10
- **Cross-attention** with the text — so every frame stays faithful to the prompt
- **Adaptive LayerNorm** — so the same network behaves differently for Pocoyo vs. Maya

**Stage 4 — Making It Real** *(Toon-Adapter + Rig Export)*

The raw output gets passed through the character-specific Toon-Adapter, which:
- Clamps joint rotations to valid ranges (elbows don't bend backwards, even in cartoons)
- Adds the character's topology bias (Pocoyo's giant head shifts the center of gravity)
- Applies the character's style signature (snappy holds, exaggerated overshoot)

The final output is a vector of rig controller values — the exact numbers an animator would type into Maya to pose the character frame by frame.

---

## The Architecture

```
Text Prompt ──→ CLIP Encoder ──→ 512d embedding
                                        │
                                        ▼
Gaussian Noise ──→ Motion Transformer (8 layers) ──→ Predicted Noise
     [B, 120, 54]    │  Self-Attention (temporal)      [B, 120, 54]
                     │  Cross-Attention (text)
                     │  Adaptive LayerNorm (time + char)
                     │
Character ID ──→ Toon-Adapter ──→ condition + style + joint_limits
                                        │
                                        ▼
                              DDIM Reverse (50 steps)
                                        │
                                        ▼
                              Clean Motion Sequence
                                   [120, 18, 3]
                                        │
                                        ▼
                              Maya Rig Controllers
                           (18 joints × 3 DOF × 120 frames)
```

---

## Why This Matters

Animation studios like Animaj produce content for 240 million kids every month on YouTube. Their characters — Pocoyo, Elly, Pato, Maya the Bee — need to move in ways that feel alive, expressive, and true to their personality.

Right now, animators manually set key poses and then spend hours filling in the frames between them. Animaj has already built AI tools for parts of this workflow:

- **Sketch-to-Pose** (ResNet50) — converts storyboard drawings to 3D poses
- **Motion In-Betweening** (Bi-LSTM) — fills frames between key poses
- **Motion-to-Motion Transfer** (GNN) — transfers motion between characters

ToonMotion-Diffusion adds a new capability to this pipeline: **generating motion directly from text descriptions**, specifically designed for the non-human cartoon characters that make kids' animation unique.

The goal isn't to replace animators. It's to give them a starting point — a first draft of motion they can refine and polish, instead of starting from a blank T-pose every time.

---

## Project Structure

```
ToonMotion-Diffusion/
├── configs/                  # Character and training configs
│   ├── default.yaml          # Default hyperparameters
│   ├── pocoyo.yaml           # Pocoyo rig + style config
│   └── maya.yaml             # Maya the Bee config
├── src/
│   ├── models/
│   │   ├── diffusion.py      # DDPM/DDIM noise schedule
│   │   ├── text_encoder.py   # CLIP text encoding
│   │   ├── toon_adapter.py   # Character-specific rig adapter
│   │   ├── motion_transformer.py  # Denoising backbone
│   │   └── toonmotion.py     # Full model (training + inference)
│   ├── data/
│   │   ├── dataset.py        # PyTorch dataset
│   │   ├── preprocessing.py  # Clean, normalize, pad
│   │   ├── augmentation.py   # Mirror, time-warp, noise
│   │   ├── maya_extractor.py # Extract from Maya files
│   │   └── validation.py     # Data quality checks
│   ├── training/
│   │   ├── trainer.py        # Training loop + checkpointing
│   │   ├── losses.py         # 5 loss functions
│   │   ├── ema.py            # Exponential moving average
│   │   └── scheduler.py      # Warmup + cosine annealing
│   ├── inference/
│   │   ├── generate.py       # Single prompt generation
│   │   ├── rig_export.py     # Convert to Maya controllers
│   │   └── batch_generate.py # Batch generation
│   ├── evaluation/
│   │   ├── metrics.py        # FID, diversity, R-precision
│   │   ├── fid_score.py      # Motion FID computation
│   │   ├── motion_quality.py # Smoothness, penetration
│   │   └── ablation.py       # Ablation study runner
│   └── api/
│       ├── server.py         # FastAPI endpoint
│       └── schemas.py        # Request/response models
├── scripts/
│   ├── train.py              # python scripts/train.py
│   ├── generate.py           # python scripts/generate.py
│   ├── evaluate.py           # python scripts/evaluate.py
│   ├── extract_maya_data.py  # Extract from Maya scenes
│   └── validate_dataset.py   # Validate data quality
├── tests/                    # Unit tests (pytest)
├── Dockerfile                # GPU container for deployment
├── requirements.txt
└── setup.py
```

---

## Quick Start

**Install:**
```bash
pip install -r requirements.txt
```

**Train (synthetic data for demo):**
```bash
python scripts/train.py --config configs/default.yaml --num_synthetic 10000
```

**Generate motion:**
```bash
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Pocoyo jumps excitedly and waves" --character Pocoyo
```

**Run API:**
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## The Loss Functions

Training uses five losses, each targeting a different aspect of animation quality:

| Loss | What It Does | Weight |
|------|-------------|--------|
| **Noise MSE** | Core diffusion objective — predict the noise accurately | 1.0 |
| **Joint Limit** | Penalize impossible poses (elbows bending 360°) | 0.01 |
| **Smoothness** | Penalize jittery motion (minimize acceleration) | 0.001 |
| **Self-Penetration** | Penalize arms passing through legs | 0.005 |
| **Foot Contact** | Penalize foot sliding when feet should be planted | 0.002 |

---

## Characters Supported

| Character | Morphology | Style | Head:Body Ratio |
|-----------|-----------|-------|-----------------|
| 👦 **Pocoyo** | Giant head, tiny body | Snappy, held poses, big overshoot | ~1:1.5 |
| 🐘 **Elly** | Elephant proportions, long trunk | Gentle, flowing, wide gestures | ~1:3 |
| 🦆 **Pato** | Duck body, wide stance, small wings | Comic, wobbly, quick reactions | ~1:2 |
| 🐝 **Maya** | Insect body, wing-like arms | Bouncy, light, hovering quality | ~1:2.5 |

Each character has its own learned embedding in the Toon-Adapter, capturing these unique proportions and movement styles.

---

## Research Papers

This project builds on ideas from the following published research:

**Diffusion Models for Motion:**
- Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2023). *Human Motion Diffusion Model*. ICLR 2023. [arXiv:2209.14916](https://arxiv.org/abs/2209.14916)
- Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., & Liu, Z. (2023). *MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model*. [arXiv:2208.15001](https://arxiv.org/abs/2208.15001)

**Text-to-Motion:**
- Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., & Chen, T. (2024). *MotionGPT: Human Motion as a Foreign Language*. NeurIPS 2023. [arXiv:2306.14795](https://arxiv.org/abs/2306.14795)
- Guo, C., Zou, S., Zuo, X., Wang, S., Ji, T., Li, X., & Cheng, L. (2022). *Generating Diverse and Natural 3D Human Motions from Text*. CVPR 2022. [arXiv:2204.14109](https://arxiv.org/abs/2204.14109)

**Motion In-Betweening:**
- Harvey, F. G., Yurick, M., Nowrouzezahrai, D., & Pal, C. (2020). *Robust Motion In-betweening*. ACM SIGGRAPH 2020. [arXiv:2102.04942](https://arxiv.org/abs/2102.04942)

**Denoising Diffusion:**
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

**Text Encoding:**
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

**Graph Neural Networks for Motion:**
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE)*. NeurIPS 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

**Animaj's Published Work** (the direct inspiration for this project):
- [Sketch-to-Pose Model](https://www.animaj.com/post/animaj-sketch-to-motion-model) — ResNet50 trained on 300K paired sketch-pose samples
- [Motion In-Betweening](https://www.animaj.com/post/animaj-sketch-to-motion-workflow-part-2-2) — Bi-LSTM on 770K frames, 67% productivity gain
- [Motion-to-Motion Transfer](https://www.animaj.com/post/animaj-motion-to-motion-transfer) — GNN on character meshes, 85% time reduction

---

## What's Next

This is a research prototype. In a production setting, the next steps would be:

1. **Train on real animation data** — replace synthetic data with actual Pocoyo episodes (770K+ frames of paired motion + text annotations)
2. **Perceptual evaluation with animators** — quantitative metrics only tell part of the story. Does the motion *feel* right for each character?
3. **Maya plugin integration** — wrap the API in a Maya plugin so animators can generate motion directly inside their workflow
4. **Multi-character scenes** — generate coordinated motion for two characters interacting (Pocoyo hugging Elly, Pato chasing Maya)
5. **Audio conditioning** — extend to audio-to-motion for dialogue and music-driven animation (see [AudioPose-Sync](https://samson1402.github.io/-AudioPose-Sync/))

---

## Author

**Sameer M** — Deep Learning Engineer, Paris

Built as a research demo for the Deep Learning Research Scientist position at [Animaj](https://www.animaj.com), the next-generation kids' media company behind Pocoyo, Maya the Bee, and 22 billion annual YouTube views.

[LinkedIn](https://linkedin.com/in/sameer-m-b73376167) · [GitHub](https://github.com/SamSon1402)

---

*"The illusion of life" — that's what the old Disney animators called it. Twelve principles, drawn by hand, frame by frame. We're not replacing that craft. We're building tools so that craft can reach more kids, in more languages, on more screens, faster than ever before.*
