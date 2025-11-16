# Kannada Music Classifier

A project to classify Kannada music into different genres.

## Project Structure

```
kannada-music-classifier/
├── raw/                     # Audio dataset with subfolders for each genre
│   ├── Dance/
│   ├── Romantic/
│   ├── Devotional/
│   └── Folk/
├── data/                    # For processed CSVs or features
├── src/
│   ├── extract_features.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── inference.py
├── web/
│   ├── app.py
│   └── templates/
│       └── index.html
├── notebooks/               # Jupyter notebooks for experimentation
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # Project License
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd kannada-music-classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Add your dataset

Place your audio files (.wav, .mp3, etc.) into the `raw/` directory, organized into subfolders by genre. For example:

```
raw/Dance/song1.mp3
raw/Romantic/song2.wav
...
```

### 2. Extract Features

Run the feature extraction script to process the audio files and generate feature CSVs in the `data/` directory.

```bash
python src/extract_features.py --raw_dir raw --output_dir data
```

You can customize the extraction parameters:

-   `--chunk_duration`: Duration of audio chunks in seconds (default: 50).
-   `--sample_rate`: Sample rate for audio processing (default: 22050).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
