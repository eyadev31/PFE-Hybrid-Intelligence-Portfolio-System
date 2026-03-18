import sys
import subprocess

try:
    from PIL import Image
    from collections import Counter
except ImportError:
    print("Installing Pillow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
    from PIL import Image
    from collections import Counter

print("Extracting colors...")
img = Image.open('Gemini_Generated_Image_5h3vhl5h3vhl5h3v.png').convert('RGB')
img = img.resize((150, 150))
colors = img.getdata()
counts = Counter(colors)
print("TOP COLORS:")
for color, count in counts.most_common(15):
    print(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x} - {count}")
