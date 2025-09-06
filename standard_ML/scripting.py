import os

# creates output.txt with: x1,x2,...,x100
with open("output.txt", "w", encoding="utf-8") as f:
    line = ",".join(f"R.{i}" for i in range(1, 101))
    f.write(line + "\n")
