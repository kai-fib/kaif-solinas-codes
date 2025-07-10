import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

source_path = filedialog.askopenfilenames()
#print(source_path)
print(f"{source_path[0]}")

output_path = filedialog.askdirectory()

print(output_path)