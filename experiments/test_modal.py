import modal

app = modal.App("test-gpu")

# Define the container image with PyTorch
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")

@app.function(gpu="T4", image=image)  # Added image parameter
def test_gpu():
    import torch
    return f"GPU available: {torch.cuda.is_available()}"

@app.local_entrypoint()
def main():
    result = test_gpu.remote()
    print(result)