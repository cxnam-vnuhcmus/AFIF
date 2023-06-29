import os
import librosa

# Đường dẫn đến thư mục chứa các file âm thanh
audio_folder = '/root/Datasets/CREMA-D/Features/audios'

# Khởi tạo biến để lưu thông tin về file âm thanh ngắn nhất
shortest_duration = float('inf')  # Đặt giá trị ban đầu là vô cùng lớn
shortest_audio_file = None

# Duyệt qua các file trong thư mục
for filename in os.listdir(audio_folder):
    if filename.endswith('.wav'):  # Chỉ xử lý các file có định dạng .wav (hoặc tùy chọn định dạng khác)
        file_path = os.path.join(audio_folder, filename)
        
        # Lấy độ dài của file âm thanh
        audio, sr = librosa.load(file_path)
        audio_duration = len(audio) / sr
        
        # Kiểm tra nếu độ dài nhỏ hơn độ dài ngắn nhất hiện tại
        if audio_duration < shortest_duration:
            shortest_duration = audio_duration
            shortest_audio_file = filename

# In ra thông tin về file âm thanh ngắn nhất
print('Shortest audio file:', shortest_audio_file)
print('Duration:', shortest_duration, 'seconds')
