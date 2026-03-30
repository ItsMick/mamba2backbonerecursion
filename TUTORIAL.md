# Mamba-3 Latent Engine Tutorial

Welcome to the cutting-edge continuous state reasoning pipeline. This tutorial covers how to deploy the **Phase 11 through Phase 13 curriculum** to force a sub-1B parameter Mamba model to reason systematically in the dark, bypassing explict Chain-of-Thought token generation.

## 1. Prerequisites \& Data Synthesis
Before jumping into the Latent Forge, you must construct the specialized curriculums:
```bash
python phase12_curriculum_builder.py
```
This script generates the required spacing-hacked arithmetic sets to protect mathematical geometry from BPE string tokenization destruction.
- `phase12a_alu.jsonl`: 15,000 pure arithmetic combinations.
- `phase12b_gsm8k.jsonl`: 7,473 curated word problems stripped of CoT formatting.

## 2. Phase 12-A: The ALU Burn-in
Train the basic numeric syntax and string addition behaviors into the Mamba matrix parameters.
```bash
python phase12a_sft_trainer.py
```
**Key Feature:** The `TURING OVERRIDE` abort sequence. The model will automatically freeze and save the weights (`mamba3_p12A_alu.pt`) if the moving average training loss crosses `< 0.15` or if validation stalls. This prevents catastrophic overfitting to arithmetic arrays.

## 3. Phase 12-B: The Semantic Bridge
Link the abstract mathematics to the linguistic English parsing.
```bash
python phase12b_sft_bridge.py
```
**Key Feature:** Uses a mixed 80/10/10 Replay Buffer natively interleaving Phase 11 boolean logic gates, Phase 12-A ALU calculations, and Target-Masked English Word queries (`labels=-100` over the prose). Aborts automatically after the action space collapses.

## 4. Phase 12-C: Final Boss GRPO
Forge the mathematical continuous constraints using Group Relative Policy Optimization.
```bash
python mamba3_p12_grpo.py
```
This script utilizes an $N=8$ branch span and unrolls dynamic continuous loops. Because the Phase 12-B format floor is solid, the specific sub-branches that calculate perfect arithmetic in the dark will receive massive positive advantage distributions.

## 5. Phase 13: Conversational Re-Anchoring
Recover English generation.
```bash
python phase13_conversational_reanchoring.py
```
Because the engine is permanently target-masked during Phase 12, Phase 13 re-injects standard SFT chat sequences at a 50/50 ratio to build a structurally unified model capable of both math mapping and casual conversation.

## Running Tests
To verify the latent routing logic against the conversational english logic, run:
```bash
python test_general.py
python test_turing.py
```
