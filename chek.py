import os

for split in ['train', 'test']:
    print(f"\n {split}")
    path = os.path.join('data', split)
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            n = len(os.listdir(cls_path))
            print(f"  {split}/{cls}: {n} файлов")