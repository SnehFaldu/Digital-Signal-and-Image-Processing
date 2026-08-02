import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr
original, sr = librosa.load("Cold_Play-Yellow.wav", sr=22050, mono=True)
karaoke, _ = librosa.load("Cold_Play-Yellow_Karaoke.wav", sr=22050, mono=True)
different, _ = librosa.load("Happy_Nation.wav", sr=22050, mono=True)
min_length = min(len(original), len(karaoke), len(different))
original = original[:min_length]
karaoke = karaoke[:min_length]
different = different[:min_length]
corr_ok = pearsonr(original, karaoke)[0]
corr_od = pearsonr(original, different)[0]
corr_kd = pearsonr(karaoke, different)[0]
print("Pearson Correlation Coefficient")
print(f"Original vs Karaoke        : {corr_ok:.4f}")
print(f"Original vs Different Song : {corr_od:.4f}")
print(f"Karaoke vs Different Song  : {corr_kd:.4f}")
def normalized_cross_corr(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    corr = correlate(x, y, mode='full')
    corr = corr / len(x)
    return corr
cross_ok = normalized_cross_corr(original, karaoke)
cross_od = normalized_cross_corr(original, different)
cross_kd = normalized_cross_corr(karaoke, different)
print("\nMaximum Normalized Cross Correlation")
print(f"Original vs Karaoke        : {np.max(cross_ok):.4f}")
print(f"Original vs Different Song : {np.max(cross_od):.4f}")
print(f"Karaoke vs Different Song  : {np.max(cross_kd):.4f}")
plt.figure(figsize=(14,8))
plt.subplot(3,1,1)
plt.plot(original)
plt.title("Original Song")
plt.subplot(3,1,2)
plt.plot(karaoke,color='green')
plt.title("Karaoke Version")
plt.subplot(3,1,3)
plt.plot(different,color='red')
plt.title("Different Song")
plt.tight_layout()
plt.figure(figsize=(14,10))
plt.subplot(3,1,1)
plt.plot(cross_ok)
plt.title("Cross Correlation: Original vs Karaoke")
plt.subplot(3,1,2)
plt.plot(cross_od,color='red')
plt.title("Cross Correlation: Original vs Different Song")
plt.subplot(3,1,3)
plt.plot(cross_kd,color='green')
plt.title("Cross Correlation: Karaoke vs Different Song")
plt.tight_layout()
plt.show()