# Example run:
# python src/extract_features.py --raw_dir raw --output_dir data --chunk_duration 50

"""
Frame-level feature extraction aggregated into 50s chunks;
saves chunk-level CSV and song-level aggregated CSV.
"""
import os
import uuid
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
import logging

# --- Constants ---
SAMPLE_RATE = 22050
CHUNK_DURATION = 50  # seconds
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# --- Functions ---

def split_audio(y: np.ndarray, sr: int, chunk_duration: int = CHUNK_DURATION) -> List[Tuple[int, int, int, np.ndarray]]:
    """Splits audio into fixed-duration chunks, padding the last one if necessary.

    Args:
        y (np.ndarray): The audio time series.
        sr (int): The sample rate.
        chunk_duration (int): The duration of each chunk in seconds.

    Returns:
        List[Tuple[int, int, int, np.ndarray]]: A list of tuples, where each tuple contains
                                                 (chunk_index, start_sample, end_sample, chunk_array).
    """
    if y.size == 0:
        warnings.warn("Input audio is empty, returning empty list.")
        return []

    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for i, start in enumerate(range(0, len(y), chunk_samples)):
        end = start + chunk_samples
        chunk = y[start:end]

        if len(chunk) < chunk_samples:
            pad_width = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), 'constant')

        chunks.append((i, start, end, chunk))
    return chunks

def extract_features_from_chunk(y_chunk: np.ndarray, sr: int) -> Dict[str, float]:
    """Extracts a flat dictionary of features from a single audio chunk.

    Args:
        y_chunk (np.ndarray): The audio chunk.
        sr (int): The sample rate.

    Returns:
        Dict[str, float]: A dictionary of audio features.
    """
    features = {}
    try:
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y_chunk, hop_length=HOP_LENGTH)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # RMS energy
        rms = librosa.feature.rms(y=y_chunk, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y_chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y_chunk, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(contrast.shape[0]):
            features[f'contrast_{i}_mean'] = np.mean(contrast[i])
            features[f'contrast_{i}_std'] = np.std(contrast[i])

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(mfcc.shape[0]):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])

        # Spectral flux
        S = np.abs(librosa.stft(y_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH))
        if S.shape[1] > 1:
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            features['spectral_flux_mean'] = np.mean(flux)
            features['spectral_flux_std'] = np.std(flux)
        else:
            features['spectral_flux_mean'] = 0.0
            features['spectral_flux_std'] = 0.0

    except Exception as e:
        warnings.warn(f"Feature extraction failed for a chunk: {e}")
        
        feature_keys = ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std', 
                        'spectral_centroid_mean', 'spectral_centroid_std',
                        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                        'spectral_rolloff_mean', 'spectral_rolloff_std',
                        'spectral_flux_mean', 'spectral_flux_std']
        for i in range(7): 
            feature_keys.extend([f'contrast_{i}_mean', f'contrast_{i}_std'])
        for i in range(N_MFCC):
            feature_keys.extend([f'mfcc{i+1}_mean', f'mfcc{i+1}_std'])
        
        features = {key: 0.0 for key in feature_keys}

    return features

def process_song(file_path: str, genre: str, sr: int = SAMPLE_RATE, chunk_duration: int = CHUNK_DURATION) -> List[Dict]:
    """Processes a single song, splitting it into chunks and extracting features.

    Args:
        file_path (str): The path to the audio file.
        genre (str): The genre of the song.
        sr (int): The sample rate.
        chunk_duration (int): The duration of each chunk in seconds.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains the features for a chunk.
    """
    filename = os.path.basename(file_path)
    song_id = uuid.uuid4().hex
    chunk_rows = []

    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        chunks = split_audio(y, sr, chunk_duration)

        for chunk_index, start_sample, end_sample, y_chunk in chunks:
            features = extract_features_from_chunk(y_chunk, sr)
            
            num_frames = len(features) # Or calculate based on a specific feature array if needed

            row = {
                'filename': filename,
                'song_id': song_id,
                'genre': genre,
                'chunk_index': chunk_index,
                'start_s': start_sample / sr,
                'end_s': end_sample / sr,
                'duration_s': chunk_duration,
                'num_frames': num_frames,
            }
            row.update(features)
            chunk_rows.append(row)

    except Exception as e:
        logging.warning(f"Error processing {file_path}: {e}")
        return []

    return chunk_rows

def save_csvs(chunk_rows: List[Dict], output_dir: str = "data") -> None:
    """Saves chunk-level and aggregated song-level features to CSV files.

    Args:
        chunk_rows (List[Dict]): A list of dictionaries containing chunk features.
        output_dir (str): The directory to save the CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_df = pd.DataFrame(chunk_rows)

    # Save chunk-level features
    chunk_csv_path = os.path.join(output_dir, 'features_chunks.csv')
    chunk_df.to_csv(chunk_csv_path, index=False)
    print(f"Saved chunk-level features to {chunk_csv_path}")

    # --- Song-level aggregation ---
    metadata_cols = ['song_id', 'filename', 'genre']
    feature_cols = [col for col in chunk_df.columns if col not in metadata_cols + 
                    ['chunk_index', 'start_s', 'end_s', 'duration_s', 'num_frames']]
    
    grouped = chunk_df.groupby(metadata_cols)
    
    # Aggregations: mean and std for features, sum for duration
    song_aggs = grouped[feature_cols].agg(['mean', 'std'])
    song_aggs.columns = ['_'.join(col).strip() for col in song_aggs.columns.values]
    
    song_duration = grouped['duration_s'].sum().to_frame(name='total_duration_s')
    
    song_df = pd.concat([song_duration, song_aggs], axis=1).reset_index()

    # Save song-level features
    song_csv_path = os.path.join(output_dir, 'features_songs.csv')
    song_df.to_csv(song_csv_path, index=False)
    print(f"Saved aggregated song-level features to {song_csv_path}")

def main(raw_dir: str = "raw", output_dir: str = "data", sr: int = SAMPLE_RATE, chunk_duration: int = CHUNK_DURATION):
    """Main function to process all songs in the raw directory.

    Args:
        raw_dir (str): The directory containing the raw audio files.
        output_dir (str): The directory to save the processed data.
        sr (int): The sample rate.
        chunk_duration (int): The duration of each chunk in seconds.
    """
    all_chunk_rows = []
    
    # Find all audio files and create a list for tqdm
    filepaths = []
    for subdir, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.au', '.webm')):
                filepaths.append((subdir, file))

    # Process files with a progress bar
    for subdir, file in tqdm(filepaths, desc="Processing songs"):
        genre = os.path.basename(subdir)
        file_path = os.path.join(subdir, file)
        
        song_chunks = process_song(file_path, genre, sr, chunk_duration)
        
        if song_chunks:
            all_chunk_rows.extend(song_chunks)
            print(f"Processed: {file} | Genre: {genre} | Chunks: {len(song_chunks)}")

    if all_chunk_rows:
        save_csvs(all_chunk_rows, output_dir)
    else:
        print("No audio files were processed. Ensure your raw directory is structured correctly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files.")
    parser.add_argument("--raw_dir", type=str, default="raw", help="Directory with raw audio files.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save feature CSVs.")
    parser.add_argument("--chunk_duration", type=int, default=50, help="Duration of audio chunks in seconds.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio processing.")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCCs to extract.")
    
    args = parser.parse_args()

    # Update module-level constants
    SAMPLE_RATE = args.sample_rate
    CHUNK_DURATION = args.chunk_duration
    N_MFCC = args.n_mfcc

    main(raw_dir=args.raw_dir, output_dir=args.output_dir, sr=args.sample_rate, chunk_duration=args.chunk_duration)