# Bomberland Project

Create a virtual environment, then run:
```
pip install -r requirements.txt
```

#### Headless testing
```
PYTHONPATH=. python3 submission_kit/run_evaluation.py --num_episodes 20 --seed 45
```

#### Simple Visualizer
```
PYTHONPATH=. python3 visualizer/run_local_evaluation.py --num_episodes 20 --seed 45
```
run at Bomberland/. headless testing for fast evaluation, visualizer for visualazing the matches (same seed = same matches)

Currently:
- Random is a noobie
- Smarter vs Simple -> never ends (both are shy)
- Smarter is lol smarter than Genius
- Genius beats Simple (and ofc beats the heck out of Random)
- Others are quite still bad