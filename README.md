# TrustTag - AI-Powered Counterfeit Product Detection

🛡️ **TrustTag** is a Progressive Web Application (PWA) that uses machine learning to detect counterfeit products and authenticate genuine items. Built with Flask and trained on product image data, it provides instant AI-powered analysis through a clean, mobile-friendly interface.

## 🚀 Quick Start

### Try It Online
No installation required! Use the live demo:
**[🔗 Try TrustTag on Hugging Face Spaces](https://huggingface.co/spaces/pepsiboi/mobile_net?logs=build)**

### Run Locally
Follow these steps to run TrustTag on your computer:

#### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

#### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mobile_net
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📱 Features

### Core Functionality
- **🔍 Image Analysis**: Upload product images for instant AI authentication
- **📊 Confidence Scoring**: Get detailed probability scores for each prediction
- **🎯 Multi-Class Detection**: Identifies counterfeit, Samsung, and Oraimo products
- **📱 PWA Support**: Install as a standalone mobile app

### Technical Features
- **🌐 Offline Support**: Works without internet connection (cached UI)
- **📲 Mobile Optimized**: Responsive design for all devices
- **⚡ Fast Processing**: Real-time ML inference
- **🔒 Secure**: Local processing, no data uploads to external servers

## 🧠 Model Details

### Dataset Classes
- **Counterfeit**: Fake/inauthentic products
- **Samsung**: Genuine Samsung products  
- **Oraimo**: Genuine Oraimo products

### Feature Extraction
- **Color Histograms**: 16 bins per RGB channel (48 features)
- **Texture Analysis**: Gradient-based texture features (4 features)
- **Total Features**: 52-dimensional feature vector

### Model Architecture
- **Algorithm**: MLP Classifier (Multi-Layer Perceptron)
- **Input Size**: 64x64 RGB images
- **Preprocessing**: Normalized pixel values with feature extraction

## 📁 Project Structure

```
mobile_net/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/
│   ├── manifest.json     # PWA manifest
│   └── sw.js            # Service worker for offline support
├── templates/
│   └── index.html       # Main web interface
├── models/              # Trained ML models
│   ├── trusttag_v1.pkl
│   ├── label_encoder.pkl
│   └── feature_params.pkl
├── data/               # Dataset (raw data excluded from git)
└── scripts/            # Training and preprocessing scripts
```

## 🔧 Development

### Training the Model
```bash
# Run training script
python scripts/train_model.py

# Preprocess data
python scripts/preprocess_data.py
```

### API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Image prediction endpoint
- `GET /health` - Health check endpoint
- `GET /sw.js` - Service worker

### Prediction API Usage
```python
import requests

# Upload image for prediction
with open('product_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', 
                           files={'image': f})
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## 🌐 Deployment

### Hugging Face Spaces
The application is deployed on Hugging Face Spaces for easy access:
- **Live URL**: https://huggingface.co/spaces/pepsiboi/mobile_net
- **Docker Support**: Includes Docker configuration
- **Auto-scaling**: Handles multiple users simultaneously

### Local Production Deployment
For production deployment, consider:
1. Using Gunicorn instead of Flask development server
2. Setting up reverse proxy with Nginx
3. Configuring SSL certificates
4. Implementing rate limiting

```bash
# Production server example
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Performance Metrics

- **Accuracy**: ~95% on test dataset
- **Inference Time**: <100ms per image
- **Model Size**: <5MB
- **Supported Formats**: PNG, JPG, JPEG, WEBP
- **Max File Size**: 16MB

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

**Port already in use**
```bash
# Kill existing process on port 5000
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 5001
```

**Model loading errors**
- Ensure all model files exist in `models/` directory
- Check Python version compatibility (3.8+)
- Verify all dependencies are installed

**Service worker not registering**
- Clear browser cache
- Check browser console for errors
- Ensure serving over HTTP or localhost

## 📞 Support

For support or questions:
- Create an issue on GitHub
- Check the Hugging Face Space for live demo
- Review the API documentation above

---

**Built with ❤️ using Flask, Tailwind CSS, and Scikit-learn**
