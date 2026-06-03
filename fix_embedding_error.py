# Run this cell first to fix the embedding import error

# Step 1: Uninstall conflicting packages
!pip uninstall sentence-transformers -y
!pip uninstall transformers -y

# Step 2: Install compatible versions
!pip install sentence-transformers==3.0.1 transformers==4.41.2 -q

# Step 3: Restart the kernel
print("✓ Dependencies fixed!")
print("\n⚠️  IMPORTANT: Now restart the kernel and run all cells sequentially from the top.")
print("Go to: Runtime → Restart runtime (in Colab) OR Kernel → Restart (in Jupyter)")
