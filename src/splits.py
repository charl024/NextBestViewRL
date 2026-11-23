from pathlib import Path

from meshes import fix_header_helper, make_splits

SEEN = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
UNSEEN = ["airplane", "bench", "bookshelf", "bottle", "car", "cup", "guitar", "lamp", "piano", "vase"]

if __name__ == "__main__":
    root = Path("data/raw/ModelNet40")
    print("headers still broken:", sum(fix_header_helper(p) for p in root.rglob("*.off")))
    make_splits(root, "data/splits", SEEN, UNSEEN)
