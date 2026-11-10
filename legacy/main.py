import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# Add near the top of your file, after imports
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# If you don't need full precision
model.half()  # Convert model to float16

# Ensure we're in evaluation mode and using GPU
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def compute_log_prob(prompt, target, model, tokenizer):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(target)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Calculate log probability for target tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probs = log_probs[:, -1, target_ids].mean()
    
    return -target_log_probs.item()  # Return negative log prob for minimization

def gcg(prompt, target, model, tokenizer, iterations=20, top_k=16, batch_size=256, allow_non_ascii=False):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).to(device)
    
    # Pre-compute once
    if not allow_non_ascii:
        ascii_mask = torch.tensor([all(c < 128 for c in tokenizer.decode([i]).encode()) 
                                 for i in range(len(tokenizer))], device=device)
    target_ids = torch.tensor(tokenizer.encode(target), device=device)
    token_embeds = model.transformer.wte.weight
    
    best_overall_loss = float('inf')
    best_overall_prompt = prompt_ids.clone()
    temperature = 0.05
    
    # Reduce number of positions to optimize per iteration
    positions_per_iter = 2  # Only modify 2 positions at a time
    
    for step in range(iterations):
        # Randomly select positions to optimize
        positions = torch.randperm(len(prompt_ids))[:positions_per_iter]
        
        embeddings = model.transformer.wte(prompt_ids).unsqueeze(0)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()
        
        outputs = model(inputs_embeds=embeddings)
        log_probs = torch.nn.functional.log_softmax(outputs.logits / temperature, dim=-1)
        target_log_probs = log_probs[:, -1:, target_ids].mean()
        
        loss = -target_log_probs
        loss.backward()
        
        grad_embeds = embeddings.grad[0].detach()
        
        # Only compute similarities for selected positions
        similarities = torch.matmul(grad_embeds[positions], token_embeds.T)
        distances = -similarities
        
        if not allow_non_ascii:
            distances[:, ~ascii_mask] = float('inf')
        
        top_k_values, top_k_indices = distances.topk(k=top_k, dim=-1, largest=False)
        
        # Create smaller candidate set
        candidates = []
        for pos_idx, pos in enumerate(positions):
            candidate_base = prompt_ids.repeat(top_k, 1)
            candidate_base[:, pos] = top_k_indices[pos_idx]
            candidates.append(candidate_base)
        candidates = torch.cat(candidates, dim=0)
        
        # Evaluate all candidates at once
        with torch.no_grad():
            outputs = model(input_ids=candidates)
            log_probs = torch.nn.functional.log_softmax(outputs.logits / temperature, dim=-1)
            losses = -log_probs[:, -1, target_ids].mean(dim=-1)
        
        best_idx = losses.argmin().item()
        current_loss = losses[best_idx].item()
        
        if current_loss < best_overall_loss:
            best_overall_loss = current_loss
            best_overall_prompt = candidates[best_idx].clone()
            prompt_ids = candidates[best_idx].clone()
        
        if step % 5 == 0:  # Reduced logging frequency
            print(f"Step {step}, Loss: {best_overall_loss}")
            print("Current prompt:", tokenizer.decode(prompt_ids))
    
    return tokenizer.decode(best_overall_prompt)

def generate(model, tokenizer, prompt, max_length=100):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define initial prompts for each keyword
keywords = [
    "radagon",
    "godfrey",
    "morgott",
    "marika",
    "radahn",
]

# Test each keyword
for target_keyword in keywords:
    print(f"\nOptimizing for: {target_keyword}")
    
    # Simple initial prompt that encourages story continuation
    initial_prompt = "The legend speaks of"
    
    optimized_prompt = gcg(
        initial_prompt,
        target_keyword,
        model,
        tokenizer,
        iterations=20,     # Moderate number of iterations
        top_k=16,         # Significantly reduced
        batch_size=256,    # Smaller batch size
        allow_non_ascii=False
    )
    print(f"Optimized Prompt: {optimized_prompt}")
    
    # Test the prompt
    with torch.no_grad():
        output = generate(model, tokenizer, optimized_prompt)
        print(f"Generated output: {output}")
        print(f"Target found: {target_keyword in output}")
