import numpy as np

# This is specific to the pong environment
def img_crop(img):
    # Remove top 30 rows and bottom 12 rows (score / borders)
    return img[30:-12, :, :]

# GENERAL Atari preprocessing steps
def downsample(img):
    # Take every 2nd pixel in height and width → half resolution
    return img[::2, ::2]

def transform_reward(reward):
    # Map rewards to -1, 0, 1 (stabilize learning)
    return np.sign(reward)

def to_grayscale(img):
    # Average RGB channels → single-channel grayscale
    return np.mean(img, axis=2).astype(np.uint8)

# ✅ Normalize grayscale image to roughly [-1, 1]
def normalize_grayscale(img):
    # 0   → -1
    # 128 →  0
    # 255 → ~0.99
    return img / 128.0 - 1.0  

def process_frame(img, image_shape):
    """
    img: raw RGB frame from env (H, W, 3)
    image_shape: (height, width) after crop + downsample
    returns: (1, H, W, 1) batch of normalized grayscale image
    """
    img = img_crop(img)
    img = downsample(img)          # Crop and downsize (by 2)
    img = to_grayscale(img)        # Convert to greyscale
    img = normalize_grayscale(img) # Normalize to [-1, 1]
    
    # Reshape to (H, W, 1) and add batch axis → (1, H, W, 1)
    return np.expand_dims(img.reshape(image_shape[0], image_shape[1], 1), axis=0)

import gym

ENV_NAME = "ALE/Pong-v5"     # or "Pong-v5" if your gym version uses that
env = gym.make(ENV_NAME, render_mode="rgb_array")

def init_state(env, image_shape, stack_size):
    """
    Initialize state for Pong with stacked frames.

    env: gym Pong environment
    image_shape: (H, W) expected by process_frame
    stack_size: how many frames to stack (e.g., 4)

    returns:
      state: (stack_size, H, W)
      frame: last processed frame (H, W)
    """
    # Gym 0.26+ returns (obs, info)
    obs, info = env.reset()

    # Process first frame
    frame = process_frame(obs, image_shape)   # (1, H, W, 1)
    frame = np.squeeze(frame, axis=0)        # (H, W, 1)
    frame = np.transpose(frame, (2, 0, 1))   # (1, H, W) → (C, H, W) = (1, H, W)
    frame = frame[0]                         # (H, W) – remove channel dim

    # Stack same frame 'stack_size' times: (stack_size, H, W)
    state = np.repeat(frame[np.newaxis, ...], stack_size, axis=0)

    return state, frame
