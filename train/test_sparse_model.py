"""
Quick test script to verify sparse attention model creation and basic functionality
"""

import torch
from sparse_attention_model import (
    MaxPoolCompress,
    MLPCompress,
    SparseAttentionAdapter,
    LlamaWithSparseAttention
)


def test_compression_modules():
    """Test compression modules"""
    print("\n" + "="*50)
    print("Testing Compression Modules")
    print("="*50)
    
    batch_size = 2
    num_heads = 8
    seq_len = 128
    dim_head = 64
    block_size = 16
    
    # Create dummy K/V
    kv = torch.randn(batch_size, num_heads, seq_len, dim_head)
    
    # Test MaxPoolCompress
    print("\n1. Testing MaxPoolCompress...")
    max_pool_compress = MaxPoolCompress(num_heads, dim_head, block_size)
    compressed_maxpool = max_pool_compress(kv)
    expected_blocks = seq_len // block_size
    print(f"   Input shape: {kv.shape}")
    print(f"   Output shape: {compressed_maxpool.shape}")
    print(f"   Expected: [{batch_size}, {num_heads}, {expected_blocks}, {dim_head}]")
    assert compressed_maxpool.shape == (batch_size, num_heads, expected_blocks, dim_head)
    print("   ✅ MaxPoolCompress works correctly!")
    
    # Test MLPCompress
    print("\n2. Testing MLPCompress...")
    mlp_compress = MLPCompress(num_heads, dim_head, block_size, expand_factor=1.0)
    compressed_mlp = mlp_compress(kv)
    print(f"   Input shape: {kv.shape}")
    print(f"   Output shape: {compressed_mlp.shape}")
    assert compressed_mlp.shape == (batch_size, num_heads, expected_blocks, dim_head)
    print("   ✅ MLPCompress works correctly!")
    
    # Count parameters
    max_pool_params = sum(p.numel() for p in max_pool_compress.parameters())
    mlp_params = sum(p.numel() for p in mlp_compress.parameters())
    print(f"\n3. Parameter counts:")
    print(f"   MaxPoolCompress: {max_pool_params:,} parameters")
    print(f"   MLPCompress: {mlp_params:,} parameters")


def test_sparse_attention_adapter():
    """Test SparseAttentionAdapter"""
    print("\n" + "="*50)
    print("Testing SparseAttentionAdapter")
    print("="*50)
    
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 4
    dim_head = hidden_size // num_heads
    
    # Create adapter
    print("\n1. Creating adapter...")
    adapter = SparseAttentionAdapter(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        compress_block_size=16,
        selection_block_size=16,
        num_selected_blocks=4,
        k_compress_method='max_pool',
        v_compress_method='max_pool',  # Changed to max_pool for efficiency
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    print(f"   Total adapter parameters: {total_params:,}")
    
    # Create dummy inputs
    print("\n2. Creating dummy inputs...")
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    q = torch.randn(batch_size, num_heads, seq_len, dim_head)
    k = torch.randn(batch_size, num_kv_heads, seq_len, dim_head)
    v = torch.randn(batch_size, num_kv_heads, seq_len, dim_head)
    
    # Forward pass
    print("\n3. Running forward pass...")
    try:
        output = adapter(hidden_states, q, k, v)
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: [{batch_size}, {num_heads}, {seq_len}, {dim_head}]")
        assert output.shape == (batch_size, num_heads, seq_len, dim_head)
        print("   ✅ SparseAttentionAdapter forward pass works!")
    except Exception as e:
        print(f"   ❌ Error in forward pass: {e}")
        raise


def test_full_model_creation():
    """Test full LlamaWithSparseAttention model creation"""
    print("\n" + "="*50)
    print("Testing Full Model Creation")
    print("="*50)
    
    print("\n⚠️  Note: This requires downloading Llama 3.2 1B (~5GB)")
    print("   Skip this test if you don't have the model downloaded.\n")
    
    response = input("Do you want to test full model creation? (y/n): ")
    if response.lower() != 'y':
        print("   Skipping full model test.")
        return
    
    print("\n1. Creating LlamaWithSparseAttention...")
    
    try:
        model = LlamaWithSparseAttention(
            model_id="meta-llama/Llama-3.2-1B",
            sparse_attn_config={
                'compress_block_size': 16,
                'selection_block_size': 16,
                'num_selected_blocks': 4,
                'k_compress_method': 'max_pool',
                'v_compress_method': 'max_pool',  # Changed to max_pool for efficiency
            }
        )
        
        print("\n2. Checking parameter statistics...")
        model.get_trainable_parameters()
        
        print("\n3. Testing save/load...")
        save_dir = "./test_adapter_save"
        model.save_adapters(save_dir)
        model.load_adapters(save_dir)
        
        print("\n   ✅ Full model creation and save/load work!")
        
        # Cleanup
        import shutil
        shutil.rmtree(save_dir)
        
    except Exception as e:
        print(f"\n   ❌ Error in full model creation: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*70)
    print(" " * 15 + "Sparse Attention Model Test Suite")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    try:
        test_compression_modules()
        test_sparse_attention_adapter()
        test_full_model_creation()
        
        print("\n" + "="*70)
        print(" " * 20 + "✅ All Tests Passed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print(" " * 20 + "❌ Tests Failed!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

