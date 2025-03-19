import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from model_def import Convautoenc
from utils import signal2pytorch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
audio_file = "punk_noisyaudio.wav"  # Update with the correct path
audio, samplerate = librosa.load(audio_file, mono=True, sr=None)
audio = audio / np.abs(audio).max()  # Normalize

X_train = signal2pytorch(audio).to(device)

# Initialize model
model = Convautoenc().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Add noise to input
    noisy_input = X_train + torch.randn_like(X_train) * 0.05
    Ypred = model(noisy_input)
    
    # Ensure input and output sizes match
    min_length = min(Ypred.shape[-1], X_train.shape[-1])
    X_train_trimmed = X_train[:, :, :min_length]
    Ypred_trimmed = Ypred[:, :, :min_length]
    
    loss = loss_fn(Ypred_trimmed, X_train_trimmed)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Save trained model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, "audio_autoenc.torch")

print("Model saved as 'audio_autoenc.torch'.")
