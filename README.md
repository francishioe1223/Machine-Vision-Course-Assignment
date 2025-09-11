# Apple Detection Systems: Traditional Vision vs YOLO vs Faster R-CNN Comparison

This repository contains my comprehensive coursework for the **Machine Vision** module in the Robotics MSc program at the University of Bristol. The project focuses on developing and comparing different computer vision approaches for automated apple detection and counting, ranging from conventional image processing techniques to state-of-the-art deep learning models.

## Project Overview

This assignment demonstrates practical implementation and comparative analysis of various computer vision techniques for object detection and counting:

- **Conventional Vision Methods**: Traditional image processing using OpenCV
- **YOLO-based Detection**: Multiple YOLO variants (YOLOv5, YOLOv10) with different configurations
- **Faster R-CNN**: Region-based CNN for object detection
- **Performance Analysis**: Comprehensive evaluation and comparison of all approaches

## Project Structure

```
├── README.md                           # This file
├── Conventional Vision Methods.ipynb   # Traditional image processing approach
├── train-yolov5s.py                   # YOLOv5 small model training
├── train-yolov5-asf.py                # YOLOv5 with augmentation and fine-tuning
├── yolov5s-PT_train.py                # YOLOv5 pre-trained model adaptation
├── train-yolov10s.py                  # YOLOv10 small model training
└── faster_rcnn/                       # Faster R-CNN implementation
    ├── train_rcnn.py                  # Training script for Faster R-CNN
    ├── predict_rcnn.py                # Inference script with post-processing
    ├── detection_eval.py              # Detection performance evaluation
    ├── counting_eval.py               # Apple counting accuracy evaluation
    ├── segmentation_eval.py           # Segmentation quality assessment
    ├── data/
    │   └── apple_dataset.py           # Custom dataset loader for apple images
    ├── utility/
    │   ├── engine.py                  # Training and evaluation engine
    │   ├── transforms.py              # Data augmentation transforms
    │   ├── coco_eval.py               # COCO-style evaluation metrics
    │   └── utils.py                   # Utility functions
    └── scripts/
        └── json_to_masks.py           # Annotation format conversion
```

## Methodology Overview

### 1. Conventional Vision Methods

**Objective:** Implement traditional computer vision techniques for apple detection using color-based segmentation and morphological operations.

#### Technical Implementation
- **Color Space Conversion**: RGB to HSV for better color-based segmentation
- **Color Filtering**: HSV thresholding to isolate apple regions (red/raw colors)
- **Morphological Operations**: Opening and closing operations for noise reduction
- **Contour Detection**: Finding apple boundaries using edge detection
- **Circle Detection**: Minimum enclosing circles for apple localization

#### Key Features
- **HSV Color Segmentation**: Robust color-based apple detection
- **Noise Reduction**: Gaussian blur and morphological filtering
- **Edge Detection**: Canny edge detection with dilation
- **Duplicate Removal**: Distance-based filtering to avoid multiple detections
- **Size Filtering**: Radius-based filtering for realistic apple sizes

### 2. YOLO-based Object Detection

**Objective:** Implement and compare different YOLO architectures for real-time apple detection and counting.

#### YOLOv5 Implementations
- **YOLOv5s Standard**: Basic YOLOv5 small model training
- **YOLOv5s with ASF**: Advanced augmentation and fine-tuning strategies
- **YOLOv5s Pre-trained**: Transfer learning from pre-trained weights

#### YOLOv10 Implementation
- **YOLOv10s**: Latest YOLO architecture with improved efficiency
- **Real-time Inference**: Gradio interface for interactive testing

#### Key Features
- **Transfer Learning**: Pre-trained weights for faster convergence
- **Data Augmentation**: Advanced augmentation strategies for robustness
- **Multi-scale Training**: Different input resolutions for various scenarios
- **Real-time Processing**: Optimized inference for practical applications

### 3. Faster R-CNN Implementation

**Objective:** Develop a region-based CNN approach for precise apple detection and localization.

#### Technical Implementation
- **ResNet-50 Backbone**: Feature extraction using pre-trained ResNet-50
- **Feature Pyramid Network (FPN)**: Multi-scale feature representation
- **Region Proposal Network (RPN)**: Efficient object proposal generation
- **ROI Pooling**: Region-of-interest feature extraction

#### Key Features
- **Two-stage Detection**: Separate proposal generation and classification
- **High Precision**: Superior localization accuracy compared to single-stage methods
- **Custom Dataset Integration**: Specialized apple dataset handling
- **Comprehensive Evaluation**: Multiple evaluation metrics implementation

## Dataset and Evaluation

### Dataset Characteristics
- **Apple Images**: Diverse collection of apple images in various conditions
- **Annotations**: Bounding box annotations for supervised learning
- **Data Splits**: Training, validation, and test sets for robust evaluation

### Evaluation Metrics
- **Detection Metrics**: Precision, Recall, F1-score, mAP
- **Counting Accuracy**: Absolute error in apple count estimation
- **Processing Speed**: Inference time for real-time applicability
- **Robustness**: Performance under different lighting and conditions

## Technical Skills Demonstrated

### Computer Vision Fundamentals
- **Image Processing**: Color space conversion, filtering, morphological operations
- **Feature Detection**: Edge detection, contour analysis, shape fitting
- **Segmentation**: Color-based and region-based segmentation techniques
- **Object Detection**: Traditional and deep learning-based approaches

### Deep Learning Implementation
- **PyTorch Framework**: Model implementation and training
- **Transfer Learning**: Pre-trained model adaptation and fine-tuning
- **Data Augmentation**: Advanced augmentation strategies for robustness
- **Model Optimization**: Hyperparameter tuning and performance optimization

### Software Engineering
- **Modular Design**: Clean code organization with reusable components
- **Evaluation Framework**: Comprehensive testing and validation pipeline
- **Documentation**: Well-documented code with clear functionality
- **Version Control**: Multiple model variants and experimental tracking

## Key Results Summary

| Approach | Method | Key Achievement | Performance Highlights |
|----------|--------|----------------|----------------------|
| Traditional | HSV + Morphology | Color-based detection | Fast processing, simple implementation |
| YOLOv5s | Standard training | Real-time detection | Good speed-accuracy balance |
| YOLOv5s-ASF | Advanced fine-tuning | Enhanced robustness | Improved generalization |
| YOLOv5s-PT | Pre-trained adaptation | Transfer learning | Faster convergence |
| YOLOv10s | Latest architecture | State-of-the-art efficiency | Best speed-accuracy trade-off |
| Faster R-CNN | Two-stage detection | Highest precision | Superior localization accuracy |

## Learning Outcomes

This project demonstrates mastery of:

1. **Traditional Computer Vision**: Color-based segmentation, morphological operations, contour analysis
2. **Deep Learning for Vision**: YOLO architectures, Faster R-CNN, transfer learning
3. **Model Comparison**: Systematic evaluation of different approaches
4. **Real-time Processing**: Optimization for practical deployment
5. **Dataset Handling**: Custom dataset creation and augmentation
6. **Performance Evaluation**: Comprehensive metrics and analysis
7. **Software Development**: Clean, modular, and maintainable code

## Implementation Details

### Environment Requirements
- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Ultralytics**: YOLO implementation
- **Torchvision**: Vision utilities and pre-trained models

## Performance Analysis

### Comparative Study
The project includes comprehensive comparison of all implemented methods across multiple dimensions:

- **Accuracy**: Detection precision and apple counting accuracy
- **Speed**: Inference time for real-time applications
- **Robustness**: Performance under varying conditions
- **Complexity**: Implementation and computational requirements

### Evaluation Framework
- **Quantitative Metrics**: mAP, precision, recall, F1-score
- **Qualitative Analysis**: Visual inspection of detection results
- **Statistical Analysis**: Confidence intervals and significance testing
- **Practical Considerations**: Deployment feasibility and resource requirements

## Future Enhancements

Potential improvements and extensions:
- **Ensemble Methods**: Combining multiple models for better performance
- **Real-time Optimization**: Model quantization and pruning for edge deployment
- **Multi-class Detection**: Extension to other fruit types
- **3D Detection**: Integration with depth information for volume estimation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

