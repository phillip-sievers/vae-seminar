import torch


def optimize_for_memory():
    """Apply memory optimizations"""
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)

    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

    # Set CUDA memory allocation configuration
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("🚀 Memory optimizations applied!")

    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        print(f"Current Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    optimize_for_memory()
