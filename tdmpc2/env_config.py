import os

def setup_env():
    os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
    os.environ['LAZY_LEGACY_OP'] = '0'
    os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
    os.environ['TORCH_LOGS'] = "+recompiles"