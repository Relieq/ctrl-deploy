# Tiny schedule on top of the official CTRL vehicle config, for pipeline smoke tests on one 8 GB GPU.
_base_ = ['./ctrl_veh_24e.py']

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(times=1),
    test=dict(samples_per_gpu=2),
)
runner = dict(type='EpochBasedRunner', max_epochs=1)
log_config = dict(interval=5)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=100)
