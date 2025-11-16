# -*- coding: utf-8 -*-
"""
Kannada Music Classifier - Visualization Notebook

This notebook provides visualizations for the extracted audio features and
spectral properties of Kannada music samples. It aims to help understand
class separation and the effectiveness of models like SVM.
"""

# %% [markdown]
# # 1. Setup and Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions # For SVM decision boundary
import librosa
import librosa.display
import random
import os

# Ensure plots are displayed inline
# %matplotlib inline # Uncomment this line if running in a Jupyter/IPython environment

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ### Mount Google Drive (if your data is there)
# Uncomment and run the following cell if your data is stored in Google Drive.

# %% 
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# ### Define Paths
# **IMPORTANT:** Update `features_csv_path` and `audio_files_dir` to your actual data locations.

# %% 
# Path to your features CSV file
# Example if on Google Drive: '/content/drive/My Drive/kannada-music-classifier/data/features_songs.csv'
features_csv_path = 'data/features_songs.csv' # Adjust this path as needed

# Path to your directory containing the .wav audio files
# Example if on Google Drive: '/content/drive/My Drive/kannada-music-classifier/raw/processed_wavs/'
# For this example, I'll assume a 'raw' directory with subfolders for genres,
# and the original .webm files were converted to .wav in place.
# You might need to adjust this significantly based on where your .wav files are.
audio_files_dir = 'raw/' # Adjust this path as needed

# %% [markdown]
# ### Load Data

# %% 
try:
    df = pd.read_csv(features_csv_path)
    print(f"Successfully loaded data from {features_csv_path}. Shape: {df.shape}")
    print("First 5 rows of the dataframe:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Features CSV not found at {features_csv_path}. Please check the path.")
    # Exit or handle the error appropriately
    # In a Colab notebook, you might want to stop execution or raise an error
    raise FileNotFoundError(f"Features CSV not found at {features_csv_path}")

# %% [markdown]
# # 2. Preprocessing

# %% 
def preprocess_for_visualization(df):
    """
    Preprocess the data by separating features and labels, encoding labels,
    imputing missing values, and scaling features.

    Args:
        df (pandas.DataFrame): The input data.

    Returns:
        tuple: A tuple containing scaled features (X_processed), encoded labels (y_encoded),
               the LabelEncoder instance, and the StandardScaler instance.
    """
    print("Preprocessing data for visualization...")
    # Assuming columns: song_id, filename, genre, then features
    # X: features (from column 3 onwards)
    # y: genre (column 2)
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]
    filenames = df.iloc[:, 1] # Keep filenames for audio visualization

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    print(f"Imputed {np.sum(np.isnan(X_imputed))} missing values.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded {len(np.unique(y_encoded))} unique genres: {label_encoder.classes_}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print("Features scaled.")

    return X_scaled, y_encoded, label_encoder, filenames

X_processed, y_encoded, label_encoder, filenames = preprocess_for_visualization(df)

# %% [markdown]
# # 3. Feature Space Visualization (PCA & t-SNE)

# %% [markdown]
# ### PCA (Principal Component Analysis)
# PCA is a linear dimensionality reduction technique that identifies the principal components
# (directions of maximum variance) in the data. It's good for capturing global structure.

# %% 
# PCA to 2D
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_processed)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
    hue=label_encoder.inverse_transform(y_encoded),
    palette=sns.color_palette("hsv", len(label_encoder.classes_)),
    legend='full',
    alpha=0.7
)
plt.title('PCA 2D Visualization of Audio Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Comment: This 2D PCA plot shows how linearly separable the different music genres are.
# If clusters are distinct, it suggests that linear models or models sensitive to linear
# relationships (like SVM with a linear kernel, or even RBF kernel if boundaries are curved)
# could perform well. Overlapping clusters indicate more complex decision boundaries are needed.

# %% 
# PCA to 3D
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_processed)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
    c=y_encoded,
    cmap='hsv',
    s=50,
    alpha=0.7
)
ax.set_title('PCA 3D Visualization of Audio Features')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
legend1 = ax.legend(*scatter.legend_elements(), title="Genres",
                    labels=label_encoder.classes_, bbox_to_anchor=(1.05, 1), loc=2)
ax.add_artist(legend1)
plt.tight_layout()
plt.show()

# Comment: The 3D PCA plot offers another perspective on linear separability.
# More distinct separation in 3D compared to 2D suggests that higher-dimensional
# features are important, and models that can leverage these (like SVM with RBF kernel)
# might find better boundaries.

# %% 
# t-SNE (t-Distributed Stochastic Neighbor Embedding)
# t-SNE is a non-linear dimensionality reduction technique particularly well-suited
# for visualizing high-dimensional datasets. It focuses on preserving local structures,
# making it good for revealing clusters.

# %% 
# t-SNE to 2D (can be computationally intensive for large datasets)
# Adjust perplexity and n_iter if needed for better visualization
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne_2d = tsne_2d.fit_transform(X_processed)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne_2d[:, 0], y=X_tsne_2d[:, 1],
    hue=label_encoder.inverse_transform(y_encoded),
    palette=sns.color_palette("hsv", len(label_encoder.classes_)),
    legend='full',
    alpha=0.7
)
plt.title('t-SNE 2D Visualization of Audio Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Comment: t-SNE often reveals more distinct clusters than PCA if the underlying
# structure is non-linear. Clear, well-separated clusters here would strongly
# support the idea that a non-linear classifier like SVM with an RBF kernel
# can effectively separate these classes. Overlapping clusters suggest inherent
# ambiguity or feature limitations.

# %% 
# t-SNE to 3D
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
X_tsne_3d = tsne_3d.fit_transform(X_processed)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2],
    c=y_encoded,
    cmap='hsv',
    s=50,
    alpha=0.7
)
ax.set_title('t-SNE 3D Visualization of Audio Features')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
legend1 = ax.legend(*scatter.legend_elements(), title="Genres",
                    labels=label_encoder.classes_, bbox_to_anchor=(1.05, 1), loc=2)
ax.add_artist(legend1)
plt.tight_layout()
plt.show()

# Comment: A 3D t-SNE plot can further clarify the separability of clusters.
# If classes form distinct, well-defined "blobs" in this space, it indicates
# that the features contain enough information for a powerful classifier
# like SVM (especially with a non-linear kernel) to draw effective boundaries.

# %% [markdown]
# ### Optional: SVM Decision Boundary on 2D PCA Space
# This visualization helps understand how an SVM (with RBF kernel, which performed best) 
# separates the classes in a simplified 2D feature space.

# %% 
# Train a simple SVM on the 2D PCA data for visualization
# Using a smaller C for smoother boundaries, adjust if needed
svm_2d_pca = SVC(kernel='rbf', C=1.0, random_state=42)
svm_2d_pca.fit(X_pca_2d, y_encoded)

plt.figure(figsize=(10, 8))
# Plot decision regions
plot_decision_regions(X_pca_2d, y_encoded, clf=svm_2d_pca, legend=2,
                      colors=','.join(sns.color_palette("hsv", len(label_encoder.classes_)).as_hex()))

# Plot support vectors
plt.scatter(svm_2d_pca.support_vectors_[:, 0],
            svm_2d_pca.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('SVM (RBF) Decision Boundary on PCA 2D Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# Manually create legend for genres as plot_decision_regions doesn't use label_encoder directly
handles, labels = plt.gca().get_legend_handles_labels()
# Filter out default labels from plot_decision_regions and add custom ones
new_handles = []
new_labels = []
for i, label in enumerate(label_encoder.classes_):
    new_handles.append(handles[i])
    new_labels.append(label)
plt.legend(new_handles, new_labels, title="Genres", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Comment: This plot directly illustrates the SVM's ability to create a non-linear
# decision boundary (due to the RBF kernel) to separate the classes. The margin
# (space between decision boundary and support vectors) and the location of
# support vectors (critical points for classification) are visible. If the
# boundaries effectively separate the clusters observed in PCA/t-SNE, it explains
# why SVM achieved high accuracy.

# %% [markdown]
# # 4. Spectral Properties Visualization of Audio Samples

# %% [markdown]
# This section visualizes the raw spectral characteristics of a few random audio samples.
# This helps to understand the underlying acoustic differences between genres that the
# extracted features (like MFCCs, spectral centroid, etc.) are trying to capture.

# %% 
def visualize_audio_features(audio_path, genre_label, sr=22050):
    """
    Loads an audio file and visualizes its spectrogram, mel-spectrogram, and MFCCs.

    Args:
        audio_path (str): Full path to the audio file.
        genre_label (str): The genre label for the audio file.
        sr (int): Sampling rate for loading the audio.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except FileNotFoundError:
        print(f"Audio file not found: {audio_path}. Skipping visualization for this sample.")
        return
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}. Skipping visualization.")
        return

    # Trim silence from the beginning and end
    y, _ = librosa.effects.trim(y)

    # Spectrogram
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - Genre: {genre_label} ({os.path.basename(audio_path)})')
    plt.tight_layout()

    # Mel-spectrogram
    plt.subplot(3, 1, 2)
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - Genre: {genre_label}')
    plt.tight_layout()

    # MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Typically 13 MFCCs
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC Heatmap - Genre: {genre_label}')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Select and Visualize Random Audio Samples
# We'll pick a few random samples from each genre (if available) to visualize their spectral properties.

# %% 
num_samples_per_genre = 2 # Number of random samples to visualize per genre

unique_genres = label_encoder.inverse_transform(np.unique(y_encoded))

for genre in unique_genres:
    print(f"\n--- Visualizing samples for Genre: {genre} ---")
    genre_indices = np.where(label_encoder.inverse_transform(y_encoded) == genre)[0]
    if len(genre_indices) == 0:
        print(f"No samples found for genre: {genre}")
        continue

    # Get filenames for this genre
    genre_filenames = filenames.iloc[genre_indices].tolist()

    # Select random samples
    selected_filenames = random.sample(genre_filenames, min(num_samples_per_genre, len(genre_filenames)))

    for filename in selected_filenames:
        # Construct the full path to the audio file
        # This assumes your audio files are organized in subfolders by genre,
        # or directly in the audio_files_dir. Adjust as per your actual structure.
        # Example: audio_path = os.path.join(audio_files_dir, genre, filename)
        # Example: audio_path = os.path.join(audio_files_dir, filename)
        # For this example, I'll assume the filename includes the genre folder if applicable,
        # or that the audio_files_dir is the parent of genre folders.
        # Let's try to infer the genre folder from the filename if it's not directly in audio_files_dir
        
        # A more robust way would be to have a mapping or a consistent folder structure.
        # For now, let's assume the filename is directly under a genre folder within audio_files_dir
        # e.g., audio_files_dir/Dance/song.wav
        
        # First, try to find the file directly in audio_files_dir
        full_audio_path = os.path.join(audio_files_dir, filename)
        if not os.path.exists(full_audio_path):
            # If not found, try looking in a subdirectory named after the genre
            full_audio_path = os.path.join(audio_files_dir, genre, filename)
            if not os.path.exists(full_audio_path):
                print(f"Could not find audio file for {filename} in {audio_files_dir} or {os.path.join(audio_files_dir, genre)}. Skipping.")
                continue

        visualize_audio_features(full_audio_path, genre)

# Comment: By visually inspecting spectrograms, mel-spectrograms, and MFCC heatmaps
# across different genres, we can observe patterns. For instance, some genres might
# have more distinct rhythmic patterns (visible in spectrograms), different
# energy distribution across frequencies (mel-spectrograms), or unique timbral
# characteristics (MFCCs). These visual differences correspond to the features
# that the SVM model uses to classify the music. Clearer visual distinctions
# between genres in these plots would explain the model's good performance.
