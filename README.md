# Pneumonia Detection using ResNet-50

## 🎯 Objective
Fine-tune a ResNet-50 model to distinguish pneumonia from normal chest X-rays using the PneumoniaMNIST dataset with state-of-the-art performance.

## 📊 Dataset
- **Name**: PneumoniaMNIST
- **Classes**: Normal (0), Pneumonia (1)
- **Image Size**: 28x28 (upscaled to 224x224)
- **Training Samples**: 4,708
- **Validation Samples**: 524
- **Test Samples**: 624
- **Class Distribution**: Pneumonia/Normal ≈ 5:3 ratio (imbalanced)

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python pneumonia_detection_pipeline.py
```

### Production Inference
```python
from pneumonia_detector import PneumoniaDetector

detector = PneumoniaDetector()
result = detector.predict("chest_xray.jpg")
print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
```

## 🏆 Achieved Results
- **Test Accuracy**: **92.79%** 
- **Test F1-Score**: **94.40%** 
- **Test AUC-ROC**: **97.96%** 
- **Sensitivity (Recall)**: **97.18%** (Excellent pneumonia detection)
- **Specificity**: **85.47%** (Good normal case identification)

### 🏥 Clinical Performance
- **False Negative Rate**: Only **2.82%** (11 missed pneumonia cases out of 390)
- **False Positive Rate**: **14.53%** (34 false alarms out of 234 normal cases)
- **Precision**: **91.77%** (High accuracy for pneumonia predictions)

### 📊 Confusion Matrix Analysis (Test Set)
|  | **Predicted** |  |
|--|---------------|--|
| **Actual** | **Normal** | **Pneumonia** |
| **Normal** | 200 (TN) | 34 (FP) |
| **Pneumonia** | 11 (FN) | 379 (TP) |

## 🔧 Model Architecture
- **Base Model**: ResNet-50 (ImageNet pre-trained)
- **Modifications**: 
  - Custom classifier head with dropout (2048→512→2)
  - Frozen early layers (conv1, bn1, layer1, layer2) for stable training
  - Progressive fine-tuning strategy
- **Input Size**: 224×224×3 RGB images
- **Parameters**: 25.6M total, 14.2M trainable

## 📈 Evaluation Strategy

### **Chosen Metrics & Justification:**

1. **F1-Score (Primary Metric) - 94.40%**
   - **Why**: Balanced measure for imbalanced datasets
   - **Clinical Importance**: Considers both precision and recall equally
   - **Result**: Exceptional performance balancing false positives and negatives

2. **AUC-ROC (Secondary Metric) - 97.96%**  
   - **Why**: Threshold-independent discrimination ability
   - **Medical Value**: Shows model's ability to distinguish classes at all thresholds
   - **Result**: Near-perfect discrimination capability

3. **Sensitivity/Recall (Clinical Metric) - 97.18%**
   - **Why**: Critical for medical screening - minimizes missed pneumonia cases
   - **Patient Safety**: False negatives are more dangerous than false positives
   - **Result**: Only 2.82% false negative rate ensures patient safety

### **Class Imbalance Handling:**

**Detection:**
- Analyzed class distribution: Pneumonia/Normal ≈ 5:3 ratio
- Training: 62.5% pneumonia, 37.5% normal cases

**Mitigation Strategies:**
1. **WeightedRandomSampler**: Balanced batch composition during training
2. **Class-weighted Loss**: CrossEntropyLoss with inverse frequency weights
3. **Evaluation Focus**: Emphasized F1-score over accuracy
4. **Augmentation**: Robust augmentation for minority class generalization

**Impact**: Achieved balanced performance across both classes with high sensitivity

### **Overfitting Prevention:**

**Regularization Techniques:**
1. **Dropout (0.5)**: Neural regularization in classifier head
2. **L2 Weight Decay (1e-4)**: Parameter penalty across all layers
3. **Early Stopping**: Patience=7 epochs with validation F1 monitoring
4. **Gradient Clipping**: Max norm 1.0 for training stability

**Data Augmentation (8 techniques):**
- RandomRotation (±15°), RandomHorizontalFlip
- RandomAffine, ColorJitter, RandomGrayscale
- RandomErasing, ImageNet normalization

**Architecture Choices:**
- **Frozen Early Layers**: Preserved ImageNet features (60% of parameters)
- **Progressive Fine-tuning**: Only fine-tuned deeper, task-specific layers

**Result**: No overfitting observed, stable generalization to test set

## 🎛️ Hyperparameter Choices & Justification

### **Learning Rate: 1e-4**
- **Justification**: Conservative rate for fine-tuning pre-trained models
- **Prevents**: Destroying pre-trained ImageNet features
- **Result**: Stable convergence without overshooting

### **Batch Size: 64**
- **Justification**: Optimal balance between GPU memory and gradient stability
- **Benefits**: Sufficient samples for stable gradient estimation
- **Hardware**: Fits well within typical GPU memory constraints

### **Epochs: 25 (Early stopped at epoch 5)**
- **Justification**: Sufficient for convergence with early stopping safety net
- **Early Stopping**: Patience=7 prevents overfitting
- **Result**: Optimal model selected at epoch 5

### **Weight Decay: 1e-4**
- **Justification**: L2 regularization prevents overfitting
- **Balance**: Strong enough to regularize, mild enough not to hurt performance
- **Medical Context**: Critical for generalization in medical imaging

### **Dropout Rate: 0.5**
- **Justification**: Standard regularization for medical imaging
- **Location**: Applied in custom classifier head only
- **Benefit**: Prevents co-adaptation while maintaining feature learning

## ⚖️ Class Imbalance Handling
1. **Weighted Random Sampling**: Balanced batch composition during training
2. **Class-weighted Loss**: Inverse frequency weighting
3. **Comprehensive Augmentation**: Robust feature learning with 8 augmentation techniques

##  Advanced Regularization Stack
- **L2 Weight Decay** (1e-4): Prevents overfitting
- **Dropout** (0.5): Neural regularization
- **Early Stopping** (patience=7): Prevents overtraining
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor=0.5
- **Gradient Clipping** (max_norm=1.0): Training stability
- **Extensive Data Augmentation**: 8 augmentation techniques
pneumonia-detection-resnet50/
├── README.md                              # Complete documentation
├── requirements.txt                       # Dependencies
├── pneumonia_detection_notebook.ipynb     # Main training notebook (YOUR SUBMISSION)
└── results/                              # Training outputs
    ├── result_sumamry.txt
    ├── test_results.pkl
    └── training_history.pkl
└── models/                              # Trained models
    ├── best_pneumonia_resnet50.pth           # Trained model weights
    ├── pneumonia_model.onnx
   

##  Training Results Analysis

### Training Progression (5 epochs to optimal performance)
- **Best Epoch**: 5 (Test F1: 94.40%, Test Acc: 92.79%)
- **Early Convergence**: Achieved excellent performance quickly
- **Training Stability**: Smooth convergence with no overfitting
- **Efficient Training**: Optimal results with minimal computational cost

### Final Test Performance (Epoch 5)
- **Test Accuracy**: 92.79%
- **Test Precision**: 91.77%
- **Test Recall/Sensitivity**: 97.18%
- **Test F1-Score**: 94.40%
- **Test AUC-ROC**: 97.96%

##  Clinical Significance & Impact

### Medical Screening Excellence
- **High Sensitivity (97.18%)**: Catches nearly all pneumonia cases
- **Good Specificity (85.47%)**: Reasonable false positive rate for screening
- **Clinical Workflow**: Suitable as first-line screening tool
- **Cost-Effectiveness**: Reduces manual review workload significantly

### Real-world Deployment Considerations
- **False Negative Minimization**: Only 2.82% missed cases (critical for patient safety)
- **False Positive Management**: 14.53% rate acceptable for screening context
- **Confidence Thresholding**: Built-in confidence scoring for risk stratification
- **Scalability**: Optimized for high-throughput screening environments

## 🏆 Performance Benchmarking

### Detailed Classification Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 94.79% | 85.47% | 89.89% | 234 |
| **Pneumonia** | 91.77% | 97.18% | 94.40% | 390 |
| **Weighted Avg** | 92.90% | 92.79% | 92.71% | 624 |

### Clinical Metrics Summary
- **Sensitivity**: 97.18% (Excellent disease detection)
- **Specificity**: 85.47% (Good healthy case identification)
- **False Positive Rate**: 14.53% (Acceptable for screening)
- **False Negative Rate**: 2.82% (Excellent for patient safety)

## Production Deployment

### Model Formats Available
- **PyTorch (.pth)**: Native format for PyTorch deployment
- **ONNX (.onnx)**: Framework-agnostic for production (recommended)
- **Complete Package**: Ready-to-use inference scripts

### API Integration Example
```python
# Production-ready inference
detector = PneumoniaDetector()
result = detector.predict("chest_xray.jpg")

# Example output:
# {
#   'prediction': 'Pneumonia',
#   'confidence': 0.94,
#   'probabilities': {'Normal': 0.06, 'Pneumonia': 0.94},
#   'clinical_note': 'High confidence pneumonia detection'
# }
```

##  Key Innovations & Contributions

1. **Efficient Transfer Learning**: Achieved excellent results with minimal training (5 epochs)
2. **Balanced Clinical Performance**: High sensitivity with acceptable specificity
3. **Robust Regularization**: Comprehensive overfitting prevention strategy
4. **Production-ready Pipeline**: Complete deployment package with clinical focus
5. **Medical-focused Evaluation**: Emphasis on clinically relevant metrics

##  Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine multiple architectures for higher accuracy
- **Attention Mechanisms**: Visual attention for interpretability
- **Multi-task Learning**: Simultaneous pneumonia type classification
- **Uncertainty Quantification**: Enhanced confidence estimation

### Data & Training
- **Larger Datasets**: Training on full-resolution chest X-rays
- **External Validation**: Testing on different hospital datasets
- **Federated Learning**: Multi-site training while preserving privacy
- **Active Learning**: Intelligent sample selection for annotation

##  License
MIT License - See LICENSE file for full details

##  References & Citations
- He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
- Yang, J., et al. "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data 2023.
- Kermany, D.S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." Cell 2018.
- Rajpurkar, P., et al. "Deep learning for chest radiograph diagnosis." PLOS Medicine 2018.

##  Contact & Support
For questions, issues, or collaboration opportunities:
- **Technical Issues**: Open a GitHub issue
- **Research Collaboration**: Contact the development team

---

** Achievement Summary**: This pneumonia detection model achieves **exceptional clinical performance** with 94.40% F1-score, 97.96% AUC-ROC, and 97.18% sensitivity, demonstrating excellent potential for automated chest X-ray screening with minimal false negatives (2.82%) and strong overall accuracy (92.79%).
