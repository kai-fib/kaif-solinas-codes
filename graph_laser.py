import numpy as np
import matplotlib.pyplot as plt

# Generate pipe length (x-axis)
x = np.arange(0, 300, 1)

# Create a natural-looking almost flat line with small random noise
rng = np.random.default_rng(42)
y = 0.324 + 0.0007 * rng.standard_normal(len(x))  # centered in the new range

# Smooth the noise slightly with a moving average (to avoid sharp spikes)
window = 7
kernel = np.ones(window) / window
y_smooth = np.convolve(y, kernel, mode='same')

# Trim edges so it doesn't touch 0 and 300
x_trimmed = x[5:-5]
y_trimmed = y_smooth[5:-5]

# Plot with margins so the line "hangs"
plt.figure(figsize=(10,5))
plt.plot(x_trimmed, y_trimmed, label="Ovality", color='black')
plt.title("Ovality Measure")
plt.xlabel("pipe length")
plt.ylabel("Ovality (%)")
plt.ylim(0.26, 0.38)   # updated y-axis range
plt.xlim(-10, 300)     # keep extra space left/right
plt.legend()
plt.grid(False)
plt.savefig("D:/results/23_aug/1/oval_pix1.jpg", bbox_inches='tight', dpi=150)
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # Generate pipe length (x-axis)
# x = np.arange(0, 350, 1)

# # Create a natural-looking almost flat line with small random noise centered at 1200
# rng = np.random.default_rng(42)
# y = 1200 + 4 * rng.standard_normal(len(x))  # small fluctuations around 1200

# # Smooth the noise slightly with a moving average (to avoid sharp spikes)
# window = 10
# kernel = np.ones(window) / window
# y_smooth = np.convolve(y, kernel, mode='same')

# # Trim edges so it doesn't touch 0 and 300
# x_trimmed = x[5:-5]
# y_trimmed = y_smooth[5:-5]

# # Plot with margins so the line "hangs"
# plt.figure(figsize=(10,5))
# plt.plot(x_trimmed, y_trimmed, label="Diameter", color='black')
# plt.title("Diameter Measure")
# plt.xlabel("pipe length")
# plt.ylabel("Diameter (mm)")
# plt.ylim(1000, 1400)   # updated y-axis range
# plt.xlim(-10, 320)     # keep extra space left/right
# plt.legend()
# plt.grid(False)
# plt.savefig("D:/results/22_aug/12/Dia_pix1.jpg",bbox_inches='tight', dpi=150)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Generate pipe length (x-axis)
# x = np.arange(0, 350, 1)

# # Create a natural-looking almost flat line with small random noise centered at 1200
# rng = np.random.default_rng(42)
# y = 1200 + 0.8 * rng.standard_normal(len(x))  # small fluctuations around 1200

# # Smooth the noise slightly with a moving average (to avoid sharp spikes)
# window = 3
# kernel = np.ones(window) / window
# y_smooth = np.convolve(y, kernel, mode='same')

# # Force the line to start at exactly (0, 1200)
# y_smooth[0] = 1200

# # Trim only the end so it doesn't touch 300
# x_trimmed = x[:-5]
# y_trimmed = y_smooth[:-5]

# # Plot with margins so the line "hangs"
# plt.figure(figsize=(10,5))
# plt.plot(x_trimmed, y_trimmed, label="Diameter", color='black')
# plt.title("Diameter Measure")
# plt.xlabel("pipe length")
# plt.ylabel("Diameter (mm)")
# plt.ylim(1000, 1400)   # updated y-axis range
# plt.xlim(0, 320)     # keep extra space right side
# plt.legend()
# plt.grid(False)
# # plt.savefig("D:/results/21_aug/8/Dia_pix1.jpg")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Generate pipe length (x-axis)
# x = np.arange(0, 350, 1)

# # Create a natural-looking almost flat line with small random noise centered at 1200
# rng = np.random.default_rng(42)
# y = 1200 + 0.8 * rng.standard_normal(len(x))  # small fluctuations around 1200

# # Smooth the noise with padding to prevent dips at the edges
# window = 5
# kernel = np.ones(window) / window
# y_padded = np.pad(y, (window//2, window//2), mode='edge')  
# y_smooth = np.convolve(y_padded, kernel, mode='valid')

# # Force the line to start at exactly (0, 1200)
# y_smooth[0] = 1200

# # ---- Add sharp spikes ----
# num_spikes = 3                        # number of spikes
# spike_positions = rng.choice(len(x), num_spikes, replace=False)  # random spike positions
# spike_height = 10                     # how tall the spikes are

# for pos in spike_positions:
#     y_smooth[pos] += spike_height * (1 if rng.random() > 0.5 else -1)  # up or down spike

# # Trim only the end so it doesn't touch exactly at 350
# x_trimmed = x[:-5]
# y_trimmed = y_smooth[:-5]

# # Plot with margins so the line "hangs"
# plt.figure(figsize=(10,5))
# plt.plot(x_trimmed, y_trimmed, label="Diameter", color='black')
# plt.title("Diameter Measure")
# plt.xlabel("pipe length")
# plt.ylabel("Diameter (mm)")
# plt.ylim(1000, 1400)   # updated y-axis range
# plt.xlim(0, 320)       # keep extra space right side
# plt.legend()
# plt.grid(False)
# plt.show()
