"""
Utility script to process raw .h5 CFD files into a PyG processed dataset.
Run from project root:
    python scripts/process_data.py
This will call FilmCoolingDataset.process() and write `data_gNN/processed/film_cooling_data.pt`.
"""
import os
import sys
import traceback

# Ensure project root is on sys.path when running as a script (python scripts/process_data.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    try:
        print('Starting dataset processing...')
        from src.data.data_loader import FilmCoolingDataset

        ds = FilmCoolingDataset(root='./data_gNN')
        # If processed file already exists and is valid, the constructor will have loaded it.
        # If not, call process() to create it (this will read .h5 files in data_gNN/raw/)
        try:
            # If the dataset already loaded data, notify and exit
            if getattr(ds, 'data', None) is not None:
                print('Processed dataset already available — skipping process().')
            else:
                print('No processed dataset found — running process()...')
                ds.process()
        except Exception:
            # Some dataset implementations expect process() to be called from an empty dataset instance
            print('Calling process() on a fresh FilmCoolingDataset instance...')
            FilmCoolingDataset(root='./data_gNN').process()

        print('Processing finished.')
    except Exception as e:
        print('An error occurred during processing:')
        traceback.print_exc()
        raise
