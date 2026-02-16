# Teaching Notes - Remote Sensing ML Tutorial

> **Instructor Guide**: Module-by-module teaching notes with timing, key concepts, and discussion prompts.

---

## Module 1: Data Pipeline - Satellite to Tiles ⏱️ 2 Pomodoros (50 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Understand Sentinel-2 multi-spectral imagery structure
2. Load and preprocess satellite data using rasterio
3. Implement tiling strategies for large images
4. Create PyTorch Dataset classes for geospatial data
5. Apply appropriate normalization techniques per spectral band

### Key Concepts (Slides: 5-7 slides, 10 min)

#### Slide 1: Remote Sensing Basics
- **What is remote sensing?**: Acquiring data without physical contact
- **Passive vs Active**: Sentinel-2 is passive (sunlight reflection)
- **Spatial resolution**: 10m = each pixel represents 10x10 meters on ground
- **Spectral resolution**: Number of bands (wavelengths) captured

#### Slide 2: Sentinel-2 Specifications
```
Band      Wavelength    Resolution    Common Use
------------------------------------------------
B2 (Blue)     490nm         10m       Atmospheric correction
B3 (Green)    560nm         10m       Vegetation health
B4 (Red)      665nm         10m       Chlorophyll absorption
B5 (Red Edge) 705nm         20m       Vegetation stress
B8 (NIR)      842nm         10m       Vegetation biomass
B11 (SWIR)   1610nm         20m       Soil moisture, burned areas
B12 (SWIR)   2190nm         20m       Burnt area detection
```

**Key Point**: Different bands capture different phenomena - burned areas are best detected in SWIR (Short-Wave Infrared)

#### Slide 3: Why Tiling?
- Full Sentinel-2 tile: 10,000 x 10,000 pixels (100M pixels!)
- Memory constraints: Can't load full image into GPU
- Training stability: Smaller batches, better gradient estimates
- Data augmentation: More samples from single image

**Common tile sizes**: 256x256, 512x512 (powers of 2 for GPU efficiency)

#### Slide 4: Data Formats
- **JP2 (JPEG 2000)**: Native Sentinel-2 format, compressed
- **GeoTIFF**: Industry standard, georeferenced
- **COG (Cloud Optimized GeoTIFF)**: For cloud storage access

**Python libraries**: rasterio, xarray, rioxarray

#### Slide 5: Normalization Strategies
```python
# Option 1: Per-band min-max (0-1 scale)
normalized = (band - band.min()) / (band.max() - band.min())

# Option 2: Standardization (z-score)
normalized = (band - band.mean()) / band.std()

# Option 3: Pre-computed stats (production)
# Use statistics from training set
normalized = (band - MEAN[b]) / STD[b]
```

**Best Practice**: Use pre-computed statistics for consistency

#### Slide 6: PyTorch Dataset API
```python
from torch.utils.data import Dataset, DataLoader

class RemoteSensingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tile_size=256):
        # Initialization
        pass
    
    def __len__(self):
        # Return number of samples
        pass
    
    def __getitem__(self, idx):
        # Load and return single sample
        return image, mask
```

**Key Methods**:
- `__init__`: Setup paths, transforms, caching
- `__len__`: Total number of samples (for epoch calculation)
- `__getitem__`: Load single sample (called by DataLoader)

#### Slide 7: Data Flow Visualization
```
Sentinel-2 Scene (.SAFE)
    ↓
Read Bands (rasterio)
    ↓
Stack to Array (H, W, C)
    ↓
Extract Tiles (256x256)
    ↓
Normalize (per-band)
    ↓
PyTorch Tensor (C, H, W)
    ↓
DataLoader (batching)
```

### Live Coding Walkthrough (5 min)

**Step 1**: Show file structure
```python
# src/data/dataset.py - What students see:
# TODO 1.1: Implement Sentinel-2 band loading
# TODO 1.2: Add tiling logic (256x256 patches)
# TODO 1.3: Implement normalization (per-band scaling)
# TODO 1.4: Create PyTorch Dataset class
```

**Step 2**: Walk through first TODO
- Import rasterio
- Open a sample file
- Show band structure
- Read into numpy array

**Step 3**: Discuss tiling approach
- Sliding window vs random crops
- Handling overlaps
- Edge cases (partial tiles)

**Step 4**: Mention normalization
- Show pre-computed stats file
- Explain why per-band (different scales)

### Student Exercise (15 min)

Students fill in `src/data/dataset.py` following TODO comments.

**Checkpoints**:
- [ ] TODO 1.1 complete: Can load single band
- [ ] TODO 1.2 complete: Can create tiles
- [ ] TODO 1.3 complete: Normalization works
- [ ] TODO 1.4 complete: Dataset class functional

### Common Pitfalls

1. **Band order confusion**: Sentinel-2 bands are not RGB order
   - Solution: Explicitly map B2→Blue, B3→Green, B4→Red

2. **Wrong array shape**: rasterio returns (bands, height, width)
   - PyTorch expects (channels, height, width) ✓
   - Not (height, width, channels) ✗

3. **Memory explosion**: Loading full scene
   - Solution: Use windowed reads or tile on-the-fly

4. **Normalization in-place**: Modifying original data
   - Solution: Always make copies: `band.copy()`

5. **Data type issues**: uint16 can't hold normalized floats
   - Solution: Convert to float32 before normalization

### Discussion Questions (5 min)

1. **Why do we need different spectral bands?** 
   - Answer: Different materials reflect/absorb different wavelengths
   - Vegetation: High NIR reflectance
   - Water: Low NIR reflectance
   - Burned areas: Low SWIR reflectance

2. **What tile size would you choose for your GPU?**
   - Answer: Depends on VRAM
   - 256x256: ~8GB GPU, batch size 16
   - 512x512: ~16GB GPU, batch size 8
   - Larger tiles = more context, less batch size

3. **Should we normalize per-image or use dataset statistics?**
   - Per-image: Works for inference on new scenes
   - Dataset stats: More stable during training
   - Best practice: Dataset stats for training, per-image for inference

### Test Verification (5 min)

Run tests:
```bash
pytest tests/test_module_01_data.py -v
```

**What tests check**:
- Dataset length is correct
- Sample shapes are (C, H, W)
- Values are in valid range [0, 1] or standardized
- No NaN or Inf values
- Batch collation works correctly

### Further Reading

- [Sentinel-2 User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Deep Learning for Earth Observation](https://www.mdpi.com/journal/remotesensing/special_issues/deep_learning_earth_observation)

---

## Module 2: UNet Architecture - Encoder-Decoder Design ⏱️ 2 Pomodoros (50 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Explain UNet architecture components
2. Implement encoder (contracting path) with convolutions and pooling
3. Implement decoder (expanding path) with upsampling
4. Add skip connections to preserve spatial information
5. Calculate output dimensions given input size

### Key Concepts (Slides: 6-8 slides, 10 min)

#### Slide 1: Semantic Segmentation Problem
- **Input**: Image (H, W, C)
- **Output**: Pixel-wise class labels (H, W, num_classes)
- **Challenge**: Classify every pixel while preserving spatial accuracy
- **Approach**: Encoder-decoder with skip connections

#### Slide 2: UNet Architecture Overview
```
Input (256x256x3)
    ↓
┌─────────────────────────────────────┐
│  ENCODER (Contracting Path)         │
│  - Capture context                  │
│  - Increase channels                │
│  - Decrease spatial dims            │
└─────────────────────────────────────┘
    ↓
Bottleneck (16x16x1024)
    ↓
┌─────────────────────────────────────┐
│  DECODER (Expanding Path)           │
│  - Enable localization              │
│  - Decrease channels                │
│  - Increase spatial dims            │
└─────────────────────────────────────┘
    ↓
Output (256x256xnum_classes)
```

**Invented by**: Olaf Ronneberger et al. (2015) for biomedical image segmentation

#### Slide 3: Encoder Design
```python
# Each encoder block:
Conv2d(in_channels, out_channels, 3x3) → BN → ReLU
Conv2d(out_channels, out_channels, 3x3) → BN → ReLU
MaxPool2d(2x2)  # Downsample by 2

# Channel progression:
64 → 128 → 256 → 512 → 1024
# Spatial dims:
256 → 128 → 64 → 32 → 16
```

**Purpose**: Extract hierarchical features
- Early layers: Edges, textures
- Deep layers: Object parts, semantics

#### Slide 4: Decoder Design
```python
# Each decoder block:
ConvTranspose2d(in_channels, out_channels, 2x2, stride=2)  # Upsample
Concatenate with skip connection  # Add encoder features
Conv2d(in_channels, out_channels, 3x3) → BN → ReLU
Conv2d(out_channels, out_channels, 3x3) → BN → ReLU

# Channel progression:
1024 → 512 → 256 → 128 → 64
# Spatial dims:
16 → 32 → 64 → 128 → 256
```

**Purpose**: Recover spatial resolution while using high-level features

#### Slide 5: Skip Connections (The Key Innovation!)
```
Encoder:    [64] ──────►┐
            ↓            │
           [128] ──────►│
            ↓            │  Skip Connections
           [256] ──────►│  (Preserve spatial detail)
            ↓            │
           [512] ──────►┘
            ↓
         Bottleneck
            ↓
Decoder:   [512] ◄──────┐
            ↓           │
           [256] ◄──────┤  Concatenate with
            ↓           │  encoder features
           [128] ◄──────┤
            ↓           │
           [64] ◄───────┘
```

**Why crucial?**:
- Without skips: Decoder loses fine-grained spatial info
- With skips: Localize precisely using early-layer features
- Especially important for thin structures (roads, boundaries)

#### Slide 6: Convolution Block Details
```python
class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
```

**Key Points**:
- `padding=1` maintains spatial size (3x3 conv reduces by 2 without padding)
- `BatchNorm`: Stabilizes training, allows higher learning rates
- `ReLU`: Non-linearity (inplace=True saves memory)

#### Slide 7: Output Layer
```python
# Final 1x1 convolution
self.out_conv = nn.Conv2d(64, num_classes, 1)

# Output shape: (batch, num_classes, H, W)
# For binary segmentation: num_classes = 1
# For multi-class: num_classes = N
```

**No activation here!** Loss function will apply:
- Binary: Sigmoid + BCE
- Multi-class: Softmax + CrossEntropy

#### Slide 8: Architecture Comparison
```
Architecture      Params    Memory    Accuracy
---------------------------------------------
UNet              31M       High      High
UNet++            36M       High      Higher
DeepLabV3+        62M       Higher    Similar
FPN               45M       High      High
LinkNet           12M       Medium    Good
```

**Why UNet for this tutorial?**
- Balanced performance and complexity
- Well-understood architecture
- Easy to implement and modify
- Proven for remote sensing

### Live Coding Walkthrough (5 min)

**Step 1**: Show UNet skeleton
```python
# src/models/unet.py structure:
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        # TODO 2.1-2.5: Fill in components
        pass
    
    def forward(self, x):
        # Encoder
        # Bottleneck
        # Decoder
        # Output
        pass
```

**Step 2**: Build DoubleConv first
- Explain why 2 conv layers
- Show padding calculation
- Run test to verify output shape

**Step 3**: Build encoder iteratively
- Add one level at a time
- Print shapes at each step
- Verify channel/spatial progression

**Step 4**: Show skip connection concatenation
```python
# Decoder example:
up = self.upconv(x)  # Upsample
concat = torch.cat([up, skip], dim=1)  # Concatenate along channels
out = self.conv(concat)
```

### Student Exercise (15 min)

Students implement `src/models/unet.py` following TODOs.

**Checkpoints**:
- [ ] TODO 2.1: DoubleConv block works
- [ ] TODO 2.2: Encoder stack built
- [ ] TODO 2.3: Decoder with upsampling
- [ ] TODO 2.4: Skip connections added
- [ ] TODO 2.5: Output layer produces correct shape

### Common Pitfalls

1. **Wrong concatenation dimension**: `dim=1` for channels, not 0 (batch)
   ```python
   # Wrong:
   torch.cat([up, skip], dim=0)  # Concatenates batches!
   
   # Correct:
   torch.cat([up, skip], dim=1)  # Concatenates channels
   ```

2. **Spatial size mismatch**: Encoder and decoder sizes must match
   - Use `padding=1` in convolutions
   - Use `output_padding` in transposed conv if needed
   - Verify shapes match before concatenation

3. **Channel mismatch in skip connection**:
   ```python
   # Encoder: 64 channels at level 1
   # Decoder upsampling: 128 → 64 channels
   # After concat: 64 + 64 = 128 channels (verify!)
   ```

4. **Forgetting final activation**: Don't add sigmoid/softmax in model
   - Loss function expects logits
   - Add activation during inference only

5. **Too deep for small images**: 5 levels requires 256x256 minimum
   - 256 → 128 → 64 → 32 → 16 → 8 (bottleneck)
   - For 128x128 images: use 4 levels max

### Discussion Questions (5 min)

1. **Why two convolution layers per block instead of one?**
   - Answer: Increases receptive field, better feature extraction
   - Two 3x3 convs ≈ one 5x5 conv but with more non-linearities

2. **What would happen without skip connections?**
   - Answer: Similar to autoencoder - blurry boundaries
   - Loss of fine-grained spatial information
   - Especially bad for thin structures

3. **When would you use transposed conv vs upsampling + conv?**
   - Transposed conv: Learnable upsampling, can create artifacts
   - Upsample + conv: Simpler, often more stable
   - Modern preference: Interpolation + 1x1 conv

4. **How would you adapt this for different input sizes?**
   - Answer: Adjust number of encoder/decoder levels
   - Ensure divisible by 2^n where n = number of levels

### Test Verification (5 min)

```bash
pytest tests/test_module_02_unet.py -v
```

**Tests verify**:
- Output shape matches input shape (spatial)
- Output channels match num_classes
- Model can process batch of images
- No NaN in forward pass
- Parameter count is reasonable

### Further Reading

- [U-Net Paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [Understanding U-Net](https://towardsdatascience.com/understanding-u-net-6127b961c483)
- [Skip Connections in Deep Learning](https://towardsdatascience.com/skip-connections-in-deep-learning-2785723e)
- [Transposed Convolutions](https://distill.pub/2016/deconv-checkerboard/)

---

## Module 3: Training Loop - AMP & Checkpointing ⏱️ 2 Pomodoros (50 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Implement a complete PyTorch training loop
2. Use Automatic Mixed Precision (AMP) for faster training
3. Implement model checkpointing (save best model)
4. Add validation loop with metrics tracking
5. Setup proper logging for monitoring training progress

### Key Concepts (Slides: 7-9 slides, 10 min)

#### Slide 1: Training Loop Components
```python
# Standard training loop structure:
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            metrics.update(outputs, targets)
```

**Phases**:
- **Training**: Update weights (model.train(), gradients enabled)
- **Validation**: Evaluate only (model.eval(), no_grad())

#### Slide 2: Automatic Mixed Precision (AMP)

**Problem**: FP32 (32-bit float) uses lots of memory and is slow

**Solution**: Use FP16 (16-bit float) where possible
- **FP16**: Half memory, faster compute on Tensor Cores
- **FP32**: Full precision for critical operations

**How it works**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)  # Forward pass in FP16
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()  # Scale loss to prevent underflow
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- ~2x faster on modern GPUs (Tensor Cores)
- ~2x less memory
- Minimal accuracy loss

#### Slide 3: Loss Functions for Segmentation

**Option 1: CrossEntropy Loss**
```python
# Multi-class segmentation
criterion = nn.CrossEntropyLoss()
# Input: (N, C, H, W) logits
# Target: (N, H, W) class indices
```

**Option 2: Dice Loss** (Better for imbalanced classes!)
```python
# Handles class imbalance (e.g., small burned areas)
def dice_loss(pred, target, smooth=1.):
    pred = F.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
```

**Combined**: BCE + Dice often works best
```python
loss = bce_loss(pred, target) + dice_loss(pred, target)
```

#### Slide 4: Optimizers and Schedulers

**Adam vs AdamW**:
```python
# Adam: Includes weight decay in gradient calculation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# AdamW: Decoupled weight decay (better generalization)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

**Learning Rate Schedulers**:
```python
# ReduceLROnPlateau: Drop LR when loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# CosineAnnealing: Smooth cosine decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

# StepLR: Drop every N epochs
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)
```

#### Slide 5: Checkpointing Strategy

**What to save**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_iou': best_iou,
    'scaler_state_dict': scaler.state_dict(),  # For AMP
}
torch.save(checkpoint, 'checkpoint.pth')
```

**Strategy**:
1. **Latest checkpoint**: Always save (resume training)
2. **Best checkpoint**: Save only when validation improves
3. **Periodic**: Every N epochs (for long training)

**Resume training**:
```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

#### Slide 6: Validation Loop

**Key differences from training**:
```python
model.eval()  # Disable dropout, batch norm uses running stats
with torch.no_grad():  # No gradient computation
    for batch in val_loader:
        outputs = model(inputs)
        # Compute metrics, don't update weights
```

**Why eval() mode?**:
- BatchNorm: Use running mean/variance instead of batch statistics
- Dropout: Disabled (deterministic inference)

**Why no_grad()?**:
- Saves memory (no gradient storage)
- Faster (no autograd overhead)

#### Slide 7: Logging and Monitoring

**What to track**:
- Training loss (per batch and epoch)
- Validation loss and metrics
- Learning rate
- Epoch time
- GPU memory usage

**Tools**:
```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)

# Weights & Biases
import wandb
wandb.log({"train_loss": loss, "val_iou": iou})

# Simple logging
import logging
logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")
```

#### Slide 8: Training Best Practices

1. **Start simple**: Get basic loop working first
2. **Verify with overfitting**: Train on single batch to ~0 loss
3. **Monitor validation**: Stop when it starts increasing (overfitting)
4. **Save frequently**: Don't lose progress to crashes
5. **Use AMP**: Almost always worth it on modern GPUs
6. **Set seeds**: For reproducibility
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   ```

#### Slide 9: Common Training Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Loss not decreasing | Flat curve | Check learning rate, data normalization |
| Loss NaN | Sudden spikes | Lower learning rate, gradient clipping |
| Overfitting | Train↓ Val↑ | More data, augmentation, regularization |
| Slow training | Long epochs | AMP, smaller model, DataLoader workers |
| OOM (Out of Memory) | CUDA error | Smaller batch size, gradient accumulation |

### Live Coding Walkthrough (5 min)

**Step 1**: Show train.py skeleton
```python
# src/training/train.py - Main components:
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    # TODO 3.3: Training loop with AMP
    pass

def validate(model, loader, criterion, device):
    # TODO 3.4: Validation loop
    pass

def main():
    # TODO 3.1: Setup device
    # TODO 3.2: Initialize model, loss, optimizer
    # Training loop
    # TODO 3.5: Checkpoint saving
    # TODO 3.6: Logging
```

**Step 2**: Walk through AMP implementation
- Show autocast context manager
- Explain GradScaler purpose
- Demo memory savings

**Step 3**: Show checkpoint structure
- What to include in checkpoint dict
- How to save best model only
- How to resume from checkpoint

**Step 4**: Quick validation example
- model.eval() vs model.train()
- torch.no_grad() context
- Metric calculation

### Student Exercise (15 min)

Students implement `src/training/train.py`.

**Checkpoints**:
- [ ] TODO 3.1: Device selection (CUDA/MPS/CPU)
- [ ] TODO 3.2: Model, loss, optimizer setup
- [ ] TODO 3.3: Training loop with AMP
- [ ] TODO 3.4: Validation loop with metrics
- [ ] TODO 3.5: Save best checkpoint
- [ ] TODO 3.6: Logging implementation

### Common Pitfalls

1. **Forgetting .zero_grad()**:
   ```python
   # Wrong:
   loss.backward()
   optimizer.step()
   # Gradients accumulate!
   
   # Correct:
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

2. **Not calling .train()/.eval()**:
   - BatchNorm behavior changes between modes
   - Dropout only active in train mode
   - Always set explicitly!

3. **Forgetting scaler.update()**:
   ```python
   scaler.step(optimizer)
   scaler.update()  # Don't forget this!
   ```

4. **Checkpointing only model weights**:
   - Must save optimizer state for resume
   - Must save epoch number
   - Must save scheduler state

5. **Logging in training loop** (too verbose):
   ```python
   # Wrong: Log every batch
   for batch in loader:
       ...
       wandb.log({"loss": loss})  # Too many points!
   
   # Correct: Log every N batches or average
   if batch_idx % 10 == 0:
       wandb.log({"loss": running_loss / 10})
   ```

6. **Not handling device properly**:
   ```python
   # Wrong:
   model = model.to(device)
   output = model(inputs)  # inputs still on CPU!
   
   # Correct:
   model = model.to(device)
   inputs = inputs.to(device)
   targets = targets.to(device)
   ```

### Discussion Questions (5 min)

1. **Why use AMP instead of full FP16?**
   - Answer: Some operations need FP32 precision
   - Gradients can underflow in FP16 (fixed by scaling)
   - Loss scaling maintains numerical stability

2. **When would you use Dice loss vs CrossEntropy?**
   - CrossEntropy: Balanced classes, general purpose
   - Dice: Imbalanced classes (e.g., small burned areas)
   - Combined: Best of both worlds

3. **How do you know if learning rate is too high/low?**
   - Too high: Loss NaN or oscillates wildly
   - Too low: Loss decreases very slowly
   - Just right: Smooth decrease, converges

4. **Why validate after each epoch instead of each batch?**
   - Answer: Validation is expensive (full forward pass)
   - Training loss per batch is noisy
   - Epoch-level gives stable estimate

5. **What should you do if validation loss starts increasing?**
   - Answer: Early stopping, reduce learning rate
   - Check for overfitting (add regularization)
   - Might need more training data

### Test Verification (5 min)

```bash
pytest tests/test_module_03_training.py -v
```

**Tests verify**:
- Training loop completes without errors
- Loss decreases over epochs (sanity check)
- Checkpoints are saved correctly
- Can resume from checkpoint
- AMP works without NaN
- Validation metrics computed correctly

**Overfit Test**:
```python
# Train on single batch for 100 epochs
# Should reach near-zero loss
```

### Further Reading

- [PyTorch Mixed Precision Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Understanding AdamW](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html)
- [Dice Loss Explained](https://medium.com/@erikgaas/dice-loss-explained-c23eb dec9f96)
- [Effective Training Tips](https://github.com/vahidk/EffectivePyTorch)

---

## Module 4: Evaluation Metrics - IoU & Dice ⏱️ 1 Pomodoro (25 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Calculate Intersection over Union (IoU / Jaccard Index)
2. Calculate Dice coefficient (F1 score for segmentation)
3. Implement per-class and mean metrics
4. Handle edge cases (empty masks, single class)
5. Choose appropriate metric for their problem

### Key Concepts (Slides: 5-6 slides, 8 min)

#### Slide 1: Why Special Metrics for Segmentation?

**Problem with accuracy**:
- Image has 1M pixels
- Class "background" = 990K pixels (99%)
- Class "burned" = 10K pixels (1%)
- Model predicts all background = 99% accuracy!
- But 0% for burned areas (class we care about)

**Solution**: Metrics that account for class imbalance and spatial overlap

#### Slide 2: Intersection over Union (IoU)

**Also called**: Jaccard Index

**Definition**:
```
IoU = Intersection / Union
    = |Pred ∩ True| / |Pred ∪ True|
    = TP / (TP + FP + FN)
```

**Visual**:
```
True Mask:       Prediction:      Intersection:
┌──────────┐    ┌──────────┐    ┌──┐
│████████  │    │   ████   │    │██│
│████████  │    │  ██████  │    │██│
│████████  │    │   ████   │    └──┘
└──────────┘    └──────────┘    
                                Union = Total area covered
                                
IoU = Intersection / Union (range: 0-1)
```

**Properties**:
- 1.0 = Perfect overlap
- 0.0 = No overlap
- Penalizes both false positives and false negatives

#### Slide 3: Dice Coefficient

**Also called**: F1 Score, Sørensen-Dice coefficient

**Definition**:
```
Dice = 2 × Intersection / (|Pred| + |True|)
     = 2 × TP / (2×TP + FP + FN)
     = 2 × IoU / (1 + IoU)
```

**Relationship to IoU**:
```
Dice = 2 × IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

**Properties**:
- Always higher than IoU for same overlap
- Weighs false positives and false negatives equally
- Common in medical imaging (smooth gradients)

**Comparison**:
```
Overlap    IoU     Dice
-----------------------
100%      1.00    1.00
 75%      0.60    0.75
 50%      0.33    0.50
 25%      0.14    0.25
```

#### Slide 4: Multi-Class Metrics

**Per-class calculation**:
```python
# For each class c:
IoU_c = Intersection_c / Union_c
Dice_c = 2 * Intersection_c / (Pred_c + True_c)

# Then aggregate:
mIoU = mean(IoU_c for all c)  # Mean IoU
fwIoU = sum(w_c * IoU_c)       # Frequency-weighted
```

**Example (3 classes)**:
```
Class      Pixels    IoU    Weight
----------------------------------
Background  950K    0.95    0.95
Vegetation   45K    0.80    0.045
Burned        5K    0.60    0.005

mIoU = (0.95 + 0.80 + 0.60) / 3 = 0.78
fwIoU = 0.95×0.95 + 0.80×0.045 + 0.60×0.005 = 0.91
```

**Which to use?**
- **mIoU**: Treats all classes equally (good for balanced)
- **fwIoU**: Weighs by frequency (closer to pixel accuracy)
- For imbalanced: Use mIoU or per-class analysis

#### Slide 5: Edge Cases and Smoothing

**Problem**: Empty masks (no positive class)
```python
# Division by zero!
IoU = Intersection / Union
# If both pred and true are empty: 0/0 = NaN
```

**Solution**: Add smoothing (epsilon)
```python
def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Empty masks: (0 + 1e-6) / (0 + 1e-6) = 1.0 ✓
```

**Other edge cases**:
- All predicted negative, some positive (FN only)
- All predicted positive, all negative (FP only)
- Smoothing handles all gracefully

#### Slide 6: Metric Selection Guide

| Scenario | Recommended Metric | Why |
|----------|-------------------|-----|
| Balanced classes | Accuracy or mIoU | All classes matter equally |
| Imbalanced classes | mIoU or Dice | Per-class performance |
| Medical imaging | Dice | Smooth gradients, clinically relevant |
| Object detection | mAP | Multiple objects, varying sizes |
| Competition/Leaderboard | mIoU | Standard benchmark |
| Production monitoring | Both IoU and Dice | Complete picture |

### Live Coding Walkthrough (3 min)

**Step 1**: Show metrics.py structure
```python
# src/evaluation/metrics.py
def iou_score(pred, target, smooth=1e-6):
    # TODO 4.1: Implement IoU
    pass

def dice_score(pred, target, smooth=1e-6):
    # TODO 4.2: Implement Dice
    pass

def multiclass_metrics(pred, target, num_classes):
    # TODO 4.3: Per-class and mean
    pass
```

**Step 2**: Quick IoU example
```python
# Binary case:
pred = torch.tensor([[1, 1, 0],
                     [1, 0, 0]])
target = torch.tensor([[1, 0, 0],
                       [1, 1, 0]])

intersection = (pred * target).sum()  # 2
union = pred.sum() + target.sum() - intersection  # 3+3-2 = 4
iou = 2 / 4  # 0.5
```

**Step 3**: Explain thresholding
```python
# Model outputs logits, need to convert to binary
pred_probs = torch.sigmoid(logits)  # For binary
pred_binary = (pred_probs > 0.5).float()

# Or for multi-class:
pred_classes = torch.argmax(logits, dim=1)
```

### Student Exercise (10 min)

Students implement `src/evaluation/metrics.py`.

**Checkpoints**:
- [ ] TODO 4.1: Binary IoU implementation
- [ ] TODO 4.2: Binary Dice implementation
- [ ] TODO 4.3: Multi-class support
- [ ] TODO 4.4: Edge case handling

### Common Pitfalls

1. **Forgetting to threshold**:
   ```python
   # Wrong:
   iou = calculate_iou(logits, target)  # logits are not binary!
   
   # Correct:
   preds = torch.argmax(logits, dim=1)  # For multi-class
   # or
   preds = (torch.sigmoid(logits) > 0.5).float()  # For binary
   ```

2. **Wrong axis for multi-class**:
   ```python
   # If predictions are (N, C, H, W):
   pred_classes = torch.argmax(pred, dim=1)  # Along channel dim
   ```

3. **Integer division (Python 2 style)**:
   ```python
   # Wrong:
   iou = intersection / union  # Returns 0 if both integers!
   
   # Correct:
   iou = intersection.float() / union.float()
   ```

4. **Confusing mean strategies**:
   - Mean of per-class IoUs ≠ IoU of all pixels
   - Both valid, but different interpretations

5. **Not handling batch dimension**:
   ```python
   # Process each sample separately, then average
   ious = []
   for i in range(batch_size):
       iou = calculate_iou(pred[i], target[i])
       ious.append(iou)
   mean_iou = sum(ious) / len(ious)
   ```

### Discussion Questions (3 min)

1. **Why is IoU always ≤ Dice?**
   - Answer: Mathematically: Dice = 2IoU/(1+IoU)
   - Since IoU ≤ 1, denominator (1+IoU) ≤ 2
   - So Dice ≥ IoU

2. **When would you get IoU = 0 but Dice > 0?**
   - Answer: Never! Dice = 0 iff IoU = 0
   - Both require some overlap

3. **Should you optimize for IoU or Dice during training?**
   - Answer: Dice is differentiable (smooth)
   - IoU is not differentiable at boundaries
   - Optimize Dice, evaluate with both

4. **What does smoothing (epsilon) actually do?**
   - Prevents division by zero
   - Gives partial credit for near-misses
   - Makes loss surface smoother

### Test Verification (5 min)

```bash
pytest tests/test_module_04_metrics.py -v
```

**Tests verify**:
- IoU calculation matches expected values
- Dice calculation matches expected values
- Relationship: Dice = 2*IoU/(1+IoU)
- Handles empty masks (returns 1.0 with smoothing)
- Multi-class metrics work correctly
- Batch processing works

**Test cases**:
- Perfect overlap (IoU=1.0, Dice=1.0)
- No overlap (IoU=0.0, Dice=0.0)
- 50% overlap (IoU=0.33, Dice=0.5)
- Empty masks (both should be 1.0 with smoothing)

### Further Reading

- [IoU vs Dice in Medical Segmentation](https://arxiv.org/abs/1806.02817)
- [Metrics for Semantic Segmentation](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)
- [Mean IoU Explained](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU)
- [Segmentation Metrics Survey](https://arxiv.org/abs/2001.05566)

---

## Module 5: ONNX Export & Optimization ⏱️ 1 Pomodoro (25 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Export trained PyTorch models to ONNX format
2. Verify ONNX model produces same outputs as PyTorch
3. Understand benefits of ONNX for deployment
4. Run inference with ONNX Runtime
5. Compare performance (speed) of PyTorch vs ONNX

### Key Concepts (Slides: 5-6 slides, 8 min)

#### Slide 1: Why Export Models?

**Training environment ≠ Production environment**:
- Training: PyTorch, GPU, research code
- Production: Various frameworks, CPU/GPU, optimized runtime

**Challenges**:
- Dependency hell (PyTorch versions)
- Slow startup (loading full PyTorch)
- Memory overhead
- Framework lock-in

**Solution**: Export to framework-agnostic format

#### Slide 2: ONNX (Open Neural Network Exchange)

**What is ONNX?**:
- Open standard for ML model representation
- Framework interoperability (PyTorch ↔ TensorFlow ↔ etc.)
- Runtime optimization (ONNX Runtime)
- Hardware acceleration (TensorRT, CoreML, etc.)

**Supported operators**: Most common layers (Conv, Linear, BN, etc.)
**Graph representation**: Computational graph in protobuf format

#### Slide 3: Export Process

```python
import torch.onnx

# 1. Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# 2. Export
torch.onnx.export(
    model,                    # PyTorch model
    dummy_input,              # Example input
    "model.onnx",             # Output path
    input_names=['input'],    # Input tensor name
    output_names=['output'],  # Output tensor name
    dynamic_axes={            # Variable batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11          # ONNX version
)
```

**Dynamic axes**: Allow variable batch sizes (not fixed to 1)

#### Slide 4: Verification

**Why verify?** ONNX conversion can change numerics slightly

```python
import onnxruntime as ort

# Load ONNX model
ort_session = ort.InferenceSession("model.onnx")

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
ort_outputs = ort_session.run(None, ort_inputs)

# Compare with PyTorch
pytorch_output = model(torch_input).detach().numpy()

# Check close
np.testing.assert_allclose(
    pytorch_output, ort_outputs[0], rtol=1e-3, atol=1e-5
)
```

**Tolerance**: Small differences expected (1e-3 to 1e-5)

#### Slide 5: ONNX Runtime

**Benefits**:
- Optimized kernels (faster than PyTorch for inference)
- Graph optimizations (constant folding, fusion)
- Hardware-specific acceleration (CUDA, TensorRT, DirectML)

**Usage**:
```python
import onnxruntime as ort

# Choose provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# Run
outputs = session.run(None, {'input': input_data})
```

**Performance tips**:
- Use batch inference when possible
- Pre-allocate output buffers
- Warm up (first inference is slower)

#### Slide 6: Optimization Options

**Level 1: Basic** (ONNX Runtime built-in):
- Constant folding
- Dead code elimination
- Operator fusion

**Level 2: Quantization** (smaller, faster):
```python
# Dynamic quantization (INT8 weights)
import onnx
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QInt8
)
# ~4x smaller, 1.5-2x faster on CPU
```

**Level 3: Hardware-specific**:
- TensorRT (NVIDIA GPUs)
- CoreML (Apple devices)
- OpenVINO (Intel)

### Live Coding Walkthrough (3 min)

**Step 1**: Show optimize.py structure
```python
# src/optimization/optimize.py
def export_to_onnx(model, dummy_input, output_path):
    # TODO 5.1: Load checkpoint
    # TODO 5.2: Export to ONNX
    pass

def verify_onnx(onnx_path, pytorch_model, test_input):
    # TODO 5.3: Verify outputs match
    pass

def benchmark(onnx_path, pytorch_model, test_input):
    # TODO 5.4: Compare inference time
    pass
```

**Step 2**: Quick export demo
```python
# Key parameters explained:
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11
)
```

**Step 3**: Verification importance
- Show example of numerical diff
- Explain acceptable tolerance

### Student Exercise (10 min)

Students implement `src/optimization/optimize.py`.

**Checkpoints**:
- [ ] TODO 5.1: Load trained checkpoint
- [ ] TODO 5.2: Export model to ONNX
- [ ] TODO 5.3: Verify ONNX outputs match PyTorch
- [ ] TODO 5.4: Benchmark inference speed

### Common Pitfalls

1. **Model in train mode during export**:
   ```python
   # Wrong:
   torch.onnx.export(model, ...)  # Dropout/BN still active!
   
   # Correct:
   model.eval()
   torch.onnx.export(model, ...)
   ```

2. **Wrong input shape**:
   ```python
   # Must match expected input
   dummy_input = torch.randn(1, 3, 256, 256)  # Not (3, 256, 256)
   ```

3. **Forgetting to move model to CPU**:
   ```python
   # ONNX export works best from CPU
   model = model.cpu()
   dummy_input = dummy_input.cpu()
   ```

4. **Not handling dynamic shapes**:
   ```python
   # Without dynamic_axes, batch size is fixed!
   dynamic_axes={
       'input': {0: 'batch', 2: 'height', 3: 'width'},
       'output': {0: 'batch', 2: 'height', 3: 'width'}
   }
   ```

5. **Numerical mismatch tolerance too strict**:
   ```python
   # Too strict:
   np.testing.assert_array_equal(pytorch_out, onnx_out)
   
   # Better:
   np.testing.assert_allclose(pytorch_out, onnx_out, rtol=1e-3)
   ```

6. **Wrong ONNX opset version**:
   - Too old: Missing operators
   - Too new: Compatibility issues
   - Safe: opset_version=11 or 13

### Discussion Questions (3 min)

1. **When would you NOT use ONNX?**
   - Answer: Research/experimentation (PyTorch easier)
   - Dynamic control flow (if/while loops)
   - Custom CUDA kernels

2. **How much speedup can you expect?**
   - Answer: 1.5-3x for CPU inference
   - Less for GPU (PyTorch already optimized)
   - More with quantization (2-4x)

3. **What's the difference between ONNX and TorchScript?**
   - TorchScript: PyTorch-specific, better Python integration
   - ONNX: Framework-agnostic, more deployment options
   - Both valid, choose based on deployment target

4. **Should you quantize during or after training?**
   - Post-training: Easier, minimal accuracy loss
   - Quantization-aware training: Better accuracy, harder
   - Start with post-training quantization

### Test Verification (5 min)

```bash
pytest tests/test_module_05_onnx.py -v
```

**Tests verify**:
- ONNX file is created
- Model can be loaded with onnxruntime
- Outputs match PyTorch within tolerance
- Inference completes without errors
- Benchmark shows speedup or comparable performance

**Test requirements**:
- Output shapes match
- Max difference < 1e-3
- Inference time measured

### Further Reading

- [ONNX Official Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [Model Optimization Guide](https://onnxruntime.ai/docs/performance/model-optimizations.html)
- [Quantization Best Practices](https://onnxruntime.ai/docs/performance/quantization.html)

---

## Module 6: FastAPI Deployment - REST API ⏱️ 2 Pomodoros (50 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Create REST API endpoints with FastAPI
2. Handle file uploads and image processing
3. Implement async request handling
4. Validate requests/responses with Pydantic
5. Structure ML model serving code

### Key Concepts (Slides: 7-8 slides, 10 min)

#### Slide 1: Why REST API for ML?

**Deployment patterns**:
- **Batch**: Process large datasets offline
- **Real-time API**: On-demand predictions via HTTP
- **Streaming**: Continuous inference (video)

**REST API benefits**:
- Language agnostic (any client can call)
- Scalable (load balancers, auto-scaling)
- Versionable (/v1/predict, /v2/predict)
- Monitorable (standard HTTP metrics)

#### Slide 2: FastAPI Overview

**Why FastAPI?**:
- **Fast**: Async support, Starlette framework
- **Type hints**: Automatic validation and docs
- **Standards**: OpenAPI, JSON Schema
- **Easy**: Minimal boilerplate

**Comparison**:
```
Framework     Speed    Async    Type Safety    Boilerplate
------------------------------------------------------------
Flask         Medium   No       No             Low
Django        Slow     Partial  No             High
FastAPI       Fast     Yes      Yes            Very Low
```

#### Slide 3: Basic FastAPI App

```python
from fastapi import FastAPI

app = FastAPI(title="Segmentation API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(image: UploadFile):
    # Process image
    # Run inference
    return {"prediction": result}
```

**Run**: `uvicorn main:app --reload`

**Auto-docs**: Visit `/docs` (Swagger UI) or `/redoc`

#### Slide 4: Request/Response Models

**Pydantic for validation**:
```python
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    image_base64: str
    threshold: float = 0.5
    
class PredictionResponse(BaseModel):
    mask: List[List[int]]
    class_probabilities: dict
    inference_time_ms: float
```

**Benefits**:
- Automatic validation (400 error if wrong type)
- Documentation generation
- IDE autocomplete

#### Slide 5: File Uploads

```python
from fastapi import File, UploadFile
from PIL import Image
import io

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    # Read file
    contents = await file.read()
    
    # Convert to PIL
    image = Image.open(io.BytesIO(contents))
    
    # Convert to tensor
    # ... preprocessing ...
    
    # Inference
    result = model(input_tensor)
    
    return {"result": result}
```

**Async**: `async/await` prevents blocking during file I/O

#### Slide 6: Loading ML Models

**Lazy loading pattern**:
```python
# Global variable (singleton)
model = None

def get_model():
    global model
    if model is None:
        model = load_onnx_model("model.onnx")
    return model

@app.post("/predict")
def predict(request: PredictionRequest):
    model = get_model()  # Loads only once
    result = model.predict(request.image)
    return result
```

**Why lazy?**:
- Fast startup (don't load on import)
- Memory efficient (load on first request)
- Better for testing (can mock)

#### Slide 7: Error Handling

```python
from fastapi import HTTPException

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Validate image
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Run inference
        result = model.predict(image)
        
        return {"prediction": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**HTTP Status Codes**:
- 200: Success
- 400: Bad request (client error)
- 422: Validation error (Pydantic)
- 500: Server error

#### Slide 8: API Design Best Practices

**Endpoints**:
```
GET  /health          - Health check
POST /predict         - Single prediction
POST /predict/batch   - Batch prediction
GET  /info            - Model metadata
GET  /docs            - API documentation (auto)
```

**Versioning**:
```
/v1/predict  - Original model
/v2/predict  - Improved model
```

**Documentation**:
- Use docstrings → auto-generates OpenAPI
- Include example requests/responses
- Document error cases

### Live Coding Walkthrough (5 min)

**Step 1**: Show main.py structure
```python
# src/api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Remote Sensing Segmentation API")

# TODO 6.1: Create app instance
# TODO 6.2: Define Pydantic models
# TODO 6.3: Model loading
# TODO 6.4: /health endpoint
# TODO 6.5: /predict endpoint
# TODO 6.6: Error handling
```

**Step 2**: Walk through request flow
```
Client Request
     ↓
FastAPI routing
     ↓
Pydantic validation
     ↓
Image preprocessing
     ↓
Model inference
     ↓
JSON response
```

**Step 3**: Show model loading
- Explain singleton pattern
- Show lazy loading benefit
- Mention thread safety

### Student Exercise (15 min)

Students implement `src/api/main.py`.

**Checkpoints**:
- [ ] TODO 6.1: FastAPI app instance with metadata
- [ ] TODO 6.2: Pydantic request/response models
- [ ] TODO 6.3: Model loading function (lazy)
- [ ] TODO 6.4: /health endpoint
- [ ] TODO 6.5: /predict endpoint with file upload
- [ ] TODO 6.6: Error handling and validation

### Common Pitfalls

1. **Synchronous model loading blocks server**:
   ```python
   # Wrong:
   model = load_model()  # Blocks startup!
   app = FastAPI()
   
   # Correct:
   model = None
   def get_model():
       global model
       if model is None:
           model = load_model()
       return model
   ```

2. **Not handling image format**:
   ```python
   # Must handle multiple formats
   if file.content_type not in ["image/jpeg", "image/png"]:
       raise HTTPException(400, "Unsupported image format")
   ```

3. **Memory leaks with large images**:
   ```python
   # Limit image size
   max_size = 4096
   if image.width > max_size or image.height > max_size:
       raise HTTPException(400, f"Image too large (max {max_size}px)")
   ```

4. **Forgetting to validate base64**:
   ```python
   import base64
   try:
       image_bytes = base64.b64decode(request.image_base64)
   except Exception:
       raise HTTPException(400, "Invalid base64 encoding")
   ```

5. **Not returning proper HTTP status codes**:
   - Always return appropriate status codes
   - Use 422 for validation errors (Pydantic does this)
   - Use 500 for unexpected server errors

6. **Blocking in async endpoint**:
   ```python
   # Wrong:
   @app.post("/predict")
   async def predict(file: UploadFile):
       result = model.predict(image)  # Blocks event loop!
   
   # Correct:
   @app.post("/predict")
   def predict(file: UploadFile):  # Remove async
       result = model.predict(image)  # Runs in thread pool
   ```

### Discussion Questions (5 min)

1. **When would you use async vs sync endpoints?**
   - Async: I/O bound (database, file system)
   - Sync: CPU bound (model inference, image processing)
   - FastAPI handles both, but choose appropriately

2. **How would you handle batch requests?**
   - Answer: Accept multiple files or list of base64
   - Process in parallel with thread pool
   - Return list of results

3. **What security concerns should you consider?**
   - Answer: File size limits, rate limiting, authentication
   - Input validation (prevent malicious images)
   - Resource limits (prevent DoS)

4. **How do you version your API?**
   - Answer: URL versioning (/v1/, /v2/)
   - Header versioning (Accept: application/vnd.api.v1+json)
   - URL is more explicit and easier

5. **Should you load model at startup or on first request?**
   - Startup: Predictable, slower start
   - First request: Faster start, first request slow
   - Middle ground: Background task to preload

### Test Verification (5 min)

```bash
pytest tests/test_module_06_api.py -v
```

**Start server manually**:
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Manual test**:
```bash
# Health check
curl http://localhost:8000/health

# Predict with file
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.png"
```

**Tests verify**:
- /health endpoint returns 200
- /predict accepts files
- Responses match Pydantic models
- Error handling works (400, 422, 500)
- Model inference produces valid output

### Further Reading

- [FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Deploying ML Models with FastAPI](https://towardsdatascience.com/deploying-ml-models-with-fastapi-4c7d7c8b9d2)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [ASGI and Uvicorn](https://www.uvicorn.org/)

---

## Module 7: Monitoring & Observability ⏱️ 1 Pomodoro (25 min)

### Learning Objectives

By the end of this module, students will be able to:
1. Add Prometheus metrics to FastAPI application
2. Track request latency, throughput, and errors
3. Implement structured logging
4. Create health check endpoints
5. Understand production monitoring concepts

### Key Concepts (Slides: 6-7 slides, 8 min)

#### Slide 1: Why Monitor ML Systems?

**ML systems fail differently than software**:
- Data drift (input distribution changes)
- Concept drift (relationship changes)
- Model degradation over time
- Silent failures (wrong predictions but no errors)

**Monitoring helps**:
- Detect issues before users complain
- Debug production problems
- Optimize performance
- Ensure SLA compliance

#### Slide 2: Key Metrics to Track

**System metrics**:
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate (5xx responses)
- Resource usage (CPU, memory, GPU)

**ML metrics**:
- Prediction confidence distribution
- Input data statistics (mean, std per feature)
- Model output distribution
- Ground truth vs predictions (if available)

**Business metrics**:
- User satisfaction
- Cost per prediction
- Cache hit rate

#### Slide 3: Prometheus Metrics Types

**Counter**: Monotonically increasing (total requests)
```python
from prometheus_client import Counter

requests_total = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
requests_total.labels(method='POST', endpoint='/predict').inc()
```

**Gauge**: Can go up or down (current memory usage)
```python
from prometheus_client import Gauge

active_requests = Gauge('active_requests', 'Currently processing')
active_requests.inc()  # Request starts
active_requests.dec()  # Request ends
```

**Histogram**: Distribution of values (latency)
```python
from prometheus_client import Histogram

request_duration = Histogram('request_duration_seconds', 'Request latency')
request_duration.observe(0.023)  # 23ms
```

**Summary**: Like histogram but quantiles calculated client-side

#### Slide 4: Instrumenting FastAPI

```python
from fastapi import Request
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start = time.time()
    
    # Count request
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    # Process request
    response = await call_next(request)
    
    # Record duration
    duration = time.time() - start
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Slide 5: Structured Logging

**Why structured?**:
- Machine readable (JSON)
- Easy to query (Elasticsearch, Splunk)
- Consistent format

```python
import logging
import json

# Configure JSON logging
logging.basicConfig(
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(request: PredictionRequest):
    logger.info(json.dumps({
        "event": "prediction_request",
        "image_size": request.image_size,
        "model_version": "1.0"
    }))
    
    result = model.predict(request.image)
    
    logger.info(json.dumps({
        "event": "prediction_complete",
        "latency_ms": latency,
        "confidence": result.confidence
    }))
    
    return result
```

#### Slide 6: Health Checks

**Liveness**: Is the app running?
```python
@app.get("/health/live")
def liveness():
    return {"status": "alive"}  # Always returns 200
```

**Readiness**: Is the app ready to serve?
```python
@app.get("/health/ready")
def readiness():
    checks = {
        "model_loaded": model is not None,
        "database_connected": db.is_connected(),
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(503, "Not ready")
```

**Why separate?**:
- Liveness: Kubernetes restarts pod if failing
- Readiness: Kubernetes stops sending traffic if failing
- Don't restart for temporary issues (DB connection)

#### Slide 7: Alerting Best Practices

**When to alert**:
- Error rate > 1% for 5 minutes
- Latency p95 > 500ms for 10 minutes
- Prediction confidence drops significantly
- Resource usage > 80%

**When NOT to alert**:
- Single failed request
- Brief latency spike
- Normal resource fluctuations

**Alert fatigue**: Too many alerts = ignored alerts

### Live Coding Walkthrough (3 min)

**Step 1**: Show what to add to main.py
```python
# src/api/main.py - Add to existing file

# TODO 7.1: Import prometheus_client
# TODO 7.2: Define metrics (Counter, Histogram)
# TODO 7.3: Add middleware for tracking
# TODO 7.4: Create /metrics endpoint
# TODO 7.5: Add structured logging
```

**Step 2**: Show middleware pattern
```python
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Start timer
    # Count request
    # Call endpoint
    # Record duration
    # Return response
```

**Step 3**: Quick logging example
- Show JSON format
- Explain why structured
- Show example log entry

### Student Exercise (10 min)

Students add monitoring to existing `src/api/main.py`.

**Checkpoints**:
- [ ] TODO 7.1: Import prometheus_client
- [ ] TODO 7.2: Track inference latency histogram
- [ ] TODO 7.3: Add request counter metric
- [ ] TODO 7.4: Implement /metrics endpoint
- [ ] TODO 7.5: Add structured logging to endpoints

### Common Pitfalls

1. **Metrics not registered**:
   ```python
   # Wrong:
   counter = Counter('name', 'help')  # Not imported properly
   
   # Correct:
   from prometheus_client import Counter
   counter = Counter('name', 'help')
   ```

2. **Too many labels (cardinality explosion)**:
   ```python
   # Wrong:
   Counter('requests', 'help', ['user_id'])  # Thousands of users!
   
   # Correct:
   Counter('requests', 'help', ['endpoint'])  # Few endpoints
   ```

3. **Forgetting to handle exceptions in middleware**:
   ```python
   @app.middleware("http")
   async def middleware(request, call_next):
       try:
           return await call_next(request)
       except Exception as e:
           error_counter.inc()  # Count errors!
           raise
   ```

4. **Not using async-safe metrics**:
   ```python
   # Wrong:
   counter += 1  # Race condition!
   
   # Correct:
   counter.inc()  # Thread-safe
   ```

5. **Logging sensitive data**:
   ```python
   # Wrong:
   logger.info(f"User {user.email} made prediction")  # PII!
   
   # Correct:
   logger.info(f"User {user_id} made prediction")  # Anonymized
   ```

6. **Blocking /metrics endpoint**:
   ```python
   # Wrong:
   @app.get("/metrics")
   async def metrics():  # Async not needed
       return generate_latest()
   
   # Correct:
   @app.get("/metrics")
   def metrics():  # Sync is fine
       return Response(generate_latest(), media_type="text/plain")
   ```

### Discussion Questions (3 min)

1. **What would you do if prediction confidence suddenly drops?**
   - Answer: Check for data drift
   - Compare input distributions
   - Consider retraining with recent data

2. **How do you monitor model performance without ground truth?**
   - Answer: Monitor input/output distributions
   - Track prediction confidence
   - A/B test with shadow deployment
   - Human-in-the-loop validation

3. **What's the difference between logs, metrics, and traces?**
   - Logs: Events (what happened when)
   - Metrics: Numbers over time (how many, how long)
   - Traces: Request flow through services

4. **How do you handle monitoring in multi-replica deployments?**
   - Answer: Prometheus scraping from each replica
   - Aggregation at Prometheus server
   - Or use pushgateway for batch jobs

### Test Verification (5 min)

```bash
pytest tests/test_module_07_monitoring.py -v
```

**Manual verification**:
```bash
# Start server
uvicorn src.api.main:app --reload

# Check metrics
curl http://localhost:8000/metrics

# Make some requests, then check again
curl http://localhost:8000/predict -F "file=@test.png"
curl http://localhost:8000/metrics
```

**Expected output**:
```
# HELP http_requests_total Total requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/predict",method="POST"} 5.0

# HELP request_duration_seconds Request duration
# TYPE request_duration_seconds histogram
request_duration_seconds_bucket{le="0.01"} 2.0
...
```

**Tests verify**:
- /metrics endpoint returns Prometheus format
- Request counter increments
- Latency histogram records values
- Logs are structured JSON
- Health checks work correctly

### Further Reading

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [FastAPI with Prometheus](https://fastapi.tiangolo.com/advanced/middleware/)
- [ML Monitoring Guide](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

---

## Appendix: Teaching Tips

### General Tips

1. **Time management**: Use a timer visible to all students
2. **Code along**: Type code live, don't just show slides
3. **Encourage questions**: Pause frequently for Q&A
4. **Show errors intentionally**: Demonstrate common mistakes
5. **Celebrate successes**: Acknowledge when tests pass

### Technical Setup

1. **Before class**:
   - Test all code examples
   - Have backup data samples
   - Prepare virtual environment
   - Test GitHub repo access

2. **During class**:
   - Keep terminal visible
   - Use large font size
   - Show file structure often

3. **After class**:
   - Share completed code
   - Provide additional resources
   - Open office hours

### Difficult Concepts

**If students struggle with**:
- **Convolutions**: Use animation/visualization
- **Backpropagation**: Simplify, focus on intuition
- **AMP**: Show memory comparison
- **Skip connections**: Draw diagram repeatedly

**Break down complex topics**:
- One concept at a time
- Use analogies (pipelines, filters)
- Connect to prior knowledge
- Provide working examples

### Assessment

**Formative (during tutorial)**:
- Do tests pass?
- Can they explain their code?
- Do they ask good questions?

**Summative (end of tutorial)**:
- Complete pipeline working?
- Can modify code for new use case?
- Understand design decisions?

---

## Quick Reference Card

### Module Checklist

| Module | Key File | Test Command | Time |
|--------|----------|--------------|------|
| 1 | `src/data/dataset.py` | `pytest tests/test_module_01_data.py -v` | 50 min |
| 2 | `src/models/unet.py` | `pytest tests/test_module_02_unet.py -v` | 50 min |
| 3 | `src/training/train.py` | `pytest tests/test_module_03_training.py -v` | 50 min |
| 4 | `src/evaluation/metrics.py` | `pytest tests/test_module_04_metrics.py -v` | 25 min |
| 5 | `src/optimization/optimize.py` | `pytest tests/test_module_05_onnx.py -v` | 25 min |
| 6 | `src/api/main.py` | `pytest tests/test_module_06_api.py -v` | 50 min |
| 7 | `src/api/main.py` | `pytest tests/test_module_07_monitoring.py -v` | 25 min |

### Common Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Development
pytest tests/ -v                    # Run all tests
pytest tests/test_module_XX.py -v   # Run specific module
uvicorn src.api.main:app --reload   # Start API

# Git
git add .
git commit -m "Module X completed"
git push origin main

# API Testing
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/predict -F "file=@image.png"
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch size, use AMP |
| Tests failing | Check TODO comments filled in |
| Import errors | Activate virtual environment |
| Module not found | Run `pip install -e .` |
| Port already in use | Use `lsof -ti:8000 \| xargs kill` |

---

*Teaching Notes Version 1.0*
*Last Updated: 2024*
