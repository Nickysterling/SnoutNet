# automate_train.py

import subprocess

def run_training(e, b, t, v, d, flip, color_jitter):
    # Base command
    command = f"python train.py --e {e} --b {b} --af {t} --vf {v} --img_dir {d}"

    # Add flags for transformations if applicable
    if flip:
        command += " --hflip"
    if color_jitter:
        command += " --color"

    # Use subprocess to execute the command
    print(f"Executing command: {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to complete

if __name__ == '__main__':
    t = r'.\data\oxford-iiit-pet-noses\train_noses.txt'
    v = r'.\data\oxford-iiit-pet-noses\test_noses.txt'
    d = r'.\data\oxford-iiit-pet-noses\images-original\images'

    run_training(e=30, b=128, t=t, v=v, d=d, flip=True, color_jitter=True)
    run_training(e=30, b=128, t=t, v=v, d=d, flip=False, color_jitter=False)
    run_training(e=30, b=128, t=t, v=v, d=d, flip=True, color_jitter=False)
    run_training(e=30, b=128, t=t, v=v, d=d, flip=False, color_jitter=True)

    run_training(e=30, b=64, t=t, v=v, d=d, flip=False, color_jitter=False)
    run_training(e=30, b=64, t=t, v=v, d=d, flip=True, color_jitter=False)
    run_training(e=30, b=64, t=t, v=v, d=d, flip=False, color_jitter=True)
    run_training(e=30, b=64, t=t, v=v, d=d, flip=True, color_jitter=True)

    run_training(e=30, b=32, t=t, v=v, d=d, flip=False, color_jitter=False)
    run_training(e=30, b=32, t=t, v=v, d=d, flip=True, color_jitter=False)
    run_training(e=30, b=32, t=t, v=v, d=d, flip=False, color_jitter=True)
    run_training(e=30, b=32, t=t, v=v, d=d, flip=True, color_jitter=True)

    run_training(e=30, b=16, t=t, v=v, d=d, flip=False, color_jitter=False)
    run_training(e=30, b=16, t=t, v=v, d=d, flip=True, color_jitter=False)
    run_training(e=30, b=16, t=t, v=v, d=d, flip=False, color_jitter=True)
    run_training(e=30, b=16, t=t, v=v, d=d, flip=True, color_jitter=True)


    