import librosa
import os
import json
from tqdm import tqdm
import warnings

# Bỏ qua tất cả cảnh báo người dùng
warnings.filterwarnings("ignore")

# Đường dẫn đến thư mục chứa các file âm thanh
audio_folder = "/root/Datasets/CREMA-D/Features/audios"

# Lặp qua các file âm thanh trong thư mục
for filename in tqdm(os.listdir(audio_folder)):
    if filename.endswith(".wav"):  # Chỉ xử lý các file có định dạng .wav
        file_path = os.path.join(audio_folder, filename)
        
        # Đọc file âm thanh
        full_audio, sr = librosa.load(file_path, sr=None)
        
        len_segment = sr // 1
        num_segment = len(full_audio) // len_segment
        
        for i in range(num_segment):
            audio = full_audio[i*len_segment:(i+1)*len_segment]
            
            # Rút trích các đặc trưng từ âm thanh
            features = {}
            
            features["filename"] = filename
            features["part"] = i
            
            # Rút trích các đặc trưng sử dụng librosa
            chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=512)
            features["chroma_stft"] = chroma_stft.ravel().tolist()
            
            chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
            features["chroma_cqt"] = chroma_cqt.ravel().tolist()
            
            chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
            features["chroma_cens"] = chroma_cens.ravel().tolist()
            
            chroma_vqt = librosa.feature.chroma_vqt(y=audio, sr=sr, intervals='ji5')
            features["chroma_vqt"] = chroma_vqt.ravel().tolist()
            
            melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512)
            features["melspectrogram"] = melspectrogram.ravel().tolist()
            
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)
            features["mfcc"] = mfcc.ravel().tolist()
            
            rms = librosa.feature.rms(y=audio)
            features["rms"] = rms.ravel().tolist()
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512)
            features["spectral_centroid"] = spectral_centroid.ravel().tolist()
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=512)
            features["spectral_bandwidth"] = spectral_bandwidth.ravel().tolist()
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=512)
            features["spectral_contrast"] = spectral_contrast.ravel().tolist()
            
            spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=512)
            features["spectral_flatness"] = spectral_flatness.ravel().tolist()
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=512)
            features["spectral_rolloff"] = spectral_rolloff.ravel().tolist()
            
            poly_features = librosa.feature.poly_features(y=audio, sr=sr, n_fft=512)
            features["poly_features"] = poly_features.ravel().tolist()
            
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features["tonnetz"] = tonnetz.ravel().tolist()
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            features["zero_crossing_rate"] = zero_crossing_rate.ravel().tolist()
            
            # Lưu đặc trưng vào file JSON
            output_file = f"/root/AFIF/Features/{filename[:-4]}_{i}.json"
            with open(output_file, "w") as f:
                json.dump(features, f)
