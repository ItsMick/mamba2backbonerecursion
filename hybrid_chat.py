import torch
import sys
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba1_engine import MODEL_ID, RecursiveMamba1_PrefixScratchpad, tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THINK_TEXT = "<THINK>"

print("======================================================================")
print("  HYBRID ENGINE: DUAL-LORA ROUTER ORCHESTRATOR")
print("======================================================================")

print("[INIT] Loading tokenizers and base backbone...")
backbone = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)

print("[INIT] Instantiating Routing Engine...")
model = RecursiveMamba1_PrefixScratchpad(backbone, lora_rank=4).to(DEVICE)
model.eval() # Ensure dropout and dark loop routing paths are fully deterministic

print("[INIT] Loading LoRA adapters into host memory...")
# We keep weights in CPU RAM to hot-swap into VRAM instantly
chat_weights = torch.load("saved_weights/mamba130m_lora_chat.pt", map_location="cpu")
rlf_weights = torch.load("saved_weights/mamba130m_v6_best.pt", map_location="cpu")

# Grab the <THINK> token ID (assume single token for simplicity, otherwise we'd sequence match)
think_token_id = tokenizer.encode(THINK_TEXT, add_special_tokens=False)[0]
eos_token_id = tokenizer.eos_token_id

def chat_generate(prompt: str, max_tokens: int = 50):
    """
    Standard autoregressive generation using the Chat adapter.
    """
    input_ids = tokenizer.encode(f"User: {prompt}\nAI: ", return_tensors="pt").to(DEVICE)
    
    generated_text = ""
    print("AI: ", end="", flush=True)
    
    with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        for _ in range(max_tokens):
            # Bypass the Recursive engine completely and generate using the base backbone + LoRA.
            # This is standard next-token prediction
            hidden_states = model.backbone(input_ids)
            logits = model.lm_head(hidden_states)
            
            # Greedy decoding for identity mapping
            next_token_id = logits[0, -1, :].argmax().item()
            
            if next_token_id == think_token_id:
                # ROUTING TRIGGER DETECTED!
                return True, generated_text
                
            if next_token_id == eos_token_id:
                break
                
            # Append and print
            token_str = tokenizer.decode([next_token_id])
            generated_text += token_str
            print(token_str, end="", flush=True)
            
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            
    return False, generated_text

def main():
    print("\\n[READY] Hybrid Engine Online. Enter your prompt (or 'exit' to quit).")
    while True:
        try:
            user_input = input("\\nUser: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input.strip():
            continue

        # 1. Engage Chat Router
        model.load_state_dict(chat_weights, strict=False)
        needs_rlf, text = chat_generate(user_input)
        
        if needs_rlf:
            print(f"\\n\\n[SYSTEM] <THINK> token intercepted. Hot-swapping to RLF Coprocessor ({len(rlf_weights)} matrices)...")
            
            # 2. Hot-Swap to Math Coprocessor
            model.load_state_dict(rlf_weights, strict=False)
            
            # Format prompt for the RLF string handler
            # Wait, the RLF expects something like: V1=123. V2=V1. What is V2? Answer:
            # We will just feed the user's raw prompt directly into the RLF assuming they typed the logic chain.
            # If they didn't, the RLF loop will naturally try its best or fail gracefully.
            if not user_input.endswith("Answer:"):
                # Append standard anchor if missing to help the bridge
                rlf_prompt = user_input.strip() + " Answer:"
            else:
                rlf_prompt = user_input.strip()
                
            input_ids = tokenizer.encode(rlf_prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                loops, trace, predicted_ans = model(input_ids, n_dark_inference=3)
                
            print(f"\\n[RLF ENGINE ACTIVATED] Executed {loops} recursive latent sweeps.")
            print(f"[RLF ENGINE TRACE] {trace}")
            print(f"\\n[SYSTEM] Restoring conversational interface...")
            
            # The RLF returns the string output token
            print(f"\\nAI: {text} <THINK> ... The answer is {predicted_ans}.")

if __name__ == "__main__":
    main()
