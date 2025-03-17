# Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

##  Overview

A real-time driver drowsiness detection system using deep learning and computer vision techniques. This system monitors a driver's eye state (open/closed) through a camera feed and alerts when signs of drowsiness are detected, helping to prevent accidents caused by driver fatigue.

![Demo Preview](assets/demo-preview.gif)

##  Key Features

- Real-time face and eye detection using OpenCV
- Advanced deep learning models for eye state classification
- Configurable alert system based on drowsiness thresholds
- Support for multiple state-of-the-art neural network architectures
- Low latency processing suitable for edge devices

##  System Architecture

The system operates in three main stages:

1. **Face & Eye Detection**: Using Haar Cascade classifiers to locate the driver's face and eyes
2. **Eye State Classification**: Applying deep learning models to determine if eyes are open or closed
3. **Drowsiness Monitoring**: Tracking eye states over time to detect drowsiness patterns

![System Architecture](https://github.com/ayoub-anhal/DRIVE_DROWSINESS/blob/main/code/Diagramme.png)

##  Technology Stack

- **Python**: Core programming language
- **OpenCV**: For image processing and Haar Cascade implementation
- **PyTorch**: Deep learning framework for model training and inference
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ayoub-anhal/DRIVE_DROWSINESS.git
   cd DRIVE_DROWSINESS
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  Dataset

The project uses a specialized dataset from [Kaggle](https://www.kaggle.com/datasets/anhalayoub/driver-drowsiness) containing images of:
- Open eyes
- Closed eyes

The dataset is balanced and preprocessed to ensure optimal model training.

##  Usage

### Training Models

To train the deep learning models on your own data:

```bash
cd code
jupyter notebook Train_models.ipynb
```

Follow the notebook instructions to:
- Load and preprocess the dataset
- Configure model hyperparameters
- Train and evaluate models
- Save the trained models

### Real-time Detection

To run the real-time drowsiness detection system:

```bash
python code/real_time.py --model resnet34  # Options: alexnet, vggnet, resnet34, custom
```

Additional parameters:
- `--camera`: Camera index (default: 0)
- `--threshold`: Drowsiness detection threshold (default: 5)
- `--alert_mode`: Alert type (sound, visual, both) (default: both)

##  Models

The system supports multiple deep learning architectures:

| Model | Accuracy | Inference Time | Size |
|-------|----------|---------------|------|
| AlexNet | 94.3% | Fast | 227 MB |
| VGGNet | 96.8% | Medium | 553 MB |
| ResNet34 | 98.2% | Medium | 83 MB |
| Custom Parallel CNN | 95.7% | Very Fast | 12 MB |

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  Contact

Ayoub Anhal - [@LinkedIn](https://www.linkedin.com/in/ayoub-anhal/)

Project Link: [https://github.com/ayoub-anhal/DRIVE_DROWSINESS](https://github.com/ayoub-anhal/DRIVE_DROWSINESS)

##  Acknowledgments

- [OpenCV Haar Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [PyTorch](https://pytorch.org/) team for the deep learning framework
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
