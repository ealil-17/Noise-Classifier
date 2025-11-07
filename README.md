# Real-time Noisy Environment Sound Classifier

A machine learning project that identifies and classifies background sounds in real-time using audio captured from a microphone or uploaded audio files. Built with TensorFlow/Keras and deployed using Streamlit for an interactive web interface.

## 🎯 Project Overview

This project uses deep learning to classify environmental sounds from the UrbanSound8K dataset. The system can identify various types of sounds including:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling
- Engine idling
- Gun shot
- Jackhammer
- Siren
- Street music

## 🏗️ Project Structure

```
Noise_Classifier/
├── Backend/
|   ├── Model_traning.py 
│   ├── model/
│   │   ├── label_encoder.pkl     # Trained label encoder
│   │   └── urbansound8k_model.h5 # Trained model
|   | 
├── Frontend/
│   ├── app.py                   # Main web application
│   ├── requirements.txt         # Python dependencies
│   └── _assets/                 # Static assets
├── virtual_env/                 # Python virtual environment
├── label_encoder.pkl           # Label encoder (copy)
├── urbansound8k_model.h5       # Trained model (copy)
│   ├── streamlit_app.py         # Main web application
└── README.md
```

## 🚀 Features

- **Real-time Audio Classification**: Classify sounds from microphone input
- **File Upload Support**: Upload and classify audio files (WAV, MP3, OGG)
- **Interactive Web Interface**: User-friendly Streamlit web application
- **MFCC Feature Extraction**: Advanced audio feature processing using librosa
- **Pre-trained Model**: Ready-to-use model trained on UrbanSound8K dataset

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ealil-17/Noise-Classifier.git
cd Noise-Classifier
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv virtual_env

# Activate virtual environment
# On Linux/Mac:
source virtual_env/bin/activate
# On Windows:
# virtual_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
cd Frontend
pip install tensorflow streamlit librosa numpy scikit-learn audio-recorder-streamlit
```

### Step 4: Verify Model Files
Ensure the following files are in the project root directory:
- `urbansound8k_model.h5` (trained model)
- `label_encoder.pkl` (label encoder)

If missing, copy them from `Backend/model/` directory.

## 🎵 How to Run

### Start the Streamlit Application
```bash
# Make sure you're in the project root directory
cd /path/to/Noise_Classifier

# Run the Streamlit app
streamlit run app.py
streamlit run Frontend/app.py
```

The application will start and open in your default web browser at `http://localhost:port`

### Using the Application

1. **Choose Input Method**:
   - **Upload Audio File**: Select a WAV, MP3, or OGG file
   - **Record Audio**: Use your microphone to record audio

2. **Audio Processing**:
   - The system extracts MFCC (Mel-frequency cepstral coefficients) features
   - Features are processed and fed to the trained neural network

3. **Get Results**:
   - View the predicted sound classification
   - Results are displayed instantly after processing

## 🧠 Model Information

- **Architecture**: Deep Neural Network with Dense layers
- **Features**: 40 MFCC coefficients
- **Training Data**: UrbanSound8K dataset
- **Audio Processing**:
  - Sample Rate: 22,050 Hz
  - Duration: 2.5 seconds (padded/truncated)
  - Feature Extraction: MFCC with 40 coefficients

## 📁 Dataset

The project uses the **UrbanSound8K** dataset:
- 8,732 labeled sound excerpts (≤4s) of urban sounds
- 10 classes of urban sounds
- Organized in 10 pre-defined folds for cross-validation
- Audio files in WAV format

## 🔧 Development

### Training a New Model
```bash
cd Backend/model
python model_training.py
```


## 📋 Requirements

Main dependencies include:
- tensorflow
- streamlit
- librosa
- numpy
- scikit-learn
- audio-recorder-streamlit



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- UrbanSound8K dataset creators
- TensorFlow and Keras teams
- Streamlit community
- librosa audio processing library

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Happy Sound Classifying! 🎵🔊**
