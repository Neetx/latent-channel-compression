from huggingface_hub import HfApi

TIERS = {
    "sequential_scaled": [
        "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
        "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
        "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
        "RecursiveMAS/Sequential-Scaled-Outerlinks",
    ],
    "mixture": [
        "RecursiveMAS/Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B",
        "RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B",
        "RecursiveMAS/Mixture-Science-BioMistral-7B",
        "RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B",
        "RecursiveMAS/Mixture-Outerlinks",
    ],
    "distillation": [
        "RecursiveMAS/Distillation-Expert-Qwen3.5-9B",
        "RecursiveMAS/Distillation-Learner-Qwen3.5-4B",
        "RecursiveMAS/Distillation-Outerlinks",
    ],
    "deliberation": [
        "RecursiveMAS/Deliberation-Reflector-Qwen3.5-4B",
        "RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B",
        "RecursiveMAS/Deliberation-Outerlinks",
    ],
}

api = HfApi()
for tier, repos in TIERS.items():
    print(f"\n=== {tier} ===")
    for r in repos:
        try:
            info = api.repo_info(repo_id=r, repo_type="model", files_metadata=True)
            size = sum((s.size or 0) for s in info.siblings) / 1e9
            print(f"  EXISTS  {r}  ({size:.2f} GB, gated={getattr(info,'gated',None)})")
        except Exception as e:
            print(f"  MISSING {r}  [{type(e).__name__}]")
