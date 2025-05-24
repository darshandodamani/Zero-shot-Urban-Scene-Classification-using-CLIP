# Zero-shot-Urban-Scene-Classification-using-CLIP

### Overview  
----  
This project implements a zero-shot classification system for urban driving scenes using OpenAI's CLIP model. By aligning image and text embeddings, it classifies scenes without the need for additional training data, facilitating efficient semantic understanding in automated driving applications.

### Motivation  
----  
Traditional scene classification models require extensive labeled datasets and retraining for new classes. This project addresses these limitations by:  
- Utilizing CLIP's pre-trained capabilities to perform classification without additional training.  
- Employing prompt engineering to define class labels dynamically.  
- Enhancing real-time inference suitable for integration with autonomous driving systems.  

### Features  
----  
- **Zero-shot Classification:** Classify images into predefined categories without additional training.  
- **Prompt Engineering:** Define and modify class labels through descriptive text prompts.  
- **Real-time Inference:** Optimized pipeline for quick processing suitable for real-time applications.  
- **Extensibility:** Modular codebase allowing easy integration of new features and classes.  

### Tech Stack  
----  
- **Python** – Core programming language  
- **PyTorch** – Deep learning framework  
- **Hugging Face Transformers** – Model loading and NLP integration  
- **OpenAI CLIP** – Vision-language model for classification  
- **CUDA** – GPU acceleration for efficient computation  
- **Matplotlib** – Visualization and debugging 