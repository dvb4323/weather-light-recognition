# Backbone-Specific Configurations

## Input Size Requirements

Different backbones have optimal input sizes:

| Backbone | Optimal Input Size | Current (Fixed) | Impact |
|----------|-------------------|-----------------|--------|
| ResNet18/34/50 | 224×224 | 224×224 ✅ | Perfect |
| EfficientNet-B0 | 224×224 | 224×224 ✅ | Perfect |
| EfficientNet-B1 | 240×240 | 224×224 ⚠️ | Slight degradation |
| EfficientNet-B2 | 260×260 | 224×224 ⚠️ | Slight degradation |
| MobileNetV3 | 224×224 | 224×224 ✅ | Perfect |

### **Current Limitation:**
All backbones use fixed 224×224 input from `config.yaml`.

### **Impact:**
- ✅ ResNet, EfficientNet-B0, MobileNet: No impact
- ⚠️ EfficientNet-B1/B2: May lose 1-2% accuracy vs optimal size

### **Future Enhancement (Optional):**
Add backbone-specific input sizes:

```python
BACKBONE_INPUT_SIZES = {
    'resnet18': 224,
    'resnet34': 224,
    'resnet50': 224,
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,  # Optimal
    'efficientnet_b2': 260,  # Optimal
    'mobilenet_v3_small': 224,
    'mobilenet_v3_large': 224,
}

input_size = BACKBONE_INPUT_SIZES.get(backbone_name, 224)
```

---

## Feature Extraction

### **Correctly Handled** ✅

Each backbone family has different feature extraction:

**ResNet:**
```python
feature_dim = model.fc.in_features  # 512 for ResNet18
```

**EfficientNet:**
```python
feature_dim = model.classifier[1].in_features  # 1280 for B0
```

**MobileNet:**
```python
feature_dim = model.classifier[0].in_features  # 960/1280
```

**Implementation**: ✅ Correctly extracts feature_dim for each backbone

---

## Output Shape Handling

### **Fixed with Adaptive Pooling** ✅

**Problem**: Different backbones may output different shapes:
- ResNet: (B, 512, 1, 1) after removing FC
- EfficientNet: (B, 1280, 7, 7) or (B, 1280)
- MobileNet: (B, 960, 1, 1) or (B, 960)

**Solution**: Added `AdaptiveAvgPool2d` in `multitask_model.py`:

```python
def forward(self, x):
    features = self.backbone(x)
    
    # Handle different output shapes
    if len(features.shape) == 4:  # (B, C, H, W)
        features = self.adaptive_pool(features)  # → (B, C, 1, 1)
    
    features = torch.flatten(features, 1)  # → (B, C)
```

This ensures all backbones output (B, feature_dim) before classification heads.

---

## Robustness Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Feature dimension** | ✅ Robust | Correctly extracted per backbone |
| **Output shape** | ✅ Robust | Adaptive pooling handles all cases |
| **Input size** | ⚠️ Fixed | 224×224 for all (optimal for most) |
| **Pretrained weights** | ✅ Robust | Correct weights API per backbone |
| **Classification heads** | ✅ Robust | Same for all backbones |

---

## Expected Performance

With current implementation (224×224 input):

| Backbone | Expected Accuracy | Notes |
|----------|------------------|-------|
| ResNet18 | 75% | Baseline ✅ |
| EfficientNet-B0 | 78-80% | Optimal ✅ |
| EfficientNet-B1 | 79-81% | Could be 80-82% with 240×240 ⚠️ |
| EfficientNet-B2 | 80-82% | Could be 81-83% with 260×260 ⚠️ |
| MobileNetV3-Large | 74-76% | Optimal ✅ |

---

## Recommendation

**Current implementation is robust enough** for experimentation:
- ✅ All backbones will work correctly
- ✅ No crashes or dimension mismatches
- ⚠️ EfficientNet-B1/B2 slightly suboptimal (1-2% loss)

**For production**: Consider adding backbone-specific input sizes to squeeze out maximum performance from EfficientNet-B1/B2.
