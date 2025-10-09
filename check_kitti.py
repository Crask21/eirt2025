from pathlib import Path

kitti_path = Path('D:/datasets/kitti')
training_path = kitti_path / 'training'

print('KITTI dataset structure:')
for p in kitti_path.iterdir():
    if p.is_dir():
        print(f'  {p.name}/')

print('\nTraining directory contents:')
for p in training_path.iterdir():
    if p.is_dir():
        file_count = len(list(p.glob('*')))
        print(f'  {p.name}/ - {file_count} files')