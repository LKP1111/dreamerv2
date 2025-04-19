from .wrappers import DMCWrapper
import os


def make_dmc_env(
        domain_name="cheetah",
        task_name="run",
        variant="video_background_camera_jitter",
        max_episode_length=1000,
        action_repeat=2,
        seed=43
):
    kwargs = dict(
        domain_name=domain_name,
        task_name=task_name,
        frame_skip=action_repeat,
        total_frames=1000,
        height=64,
        width=64,
        episode_length=max_episode_length,
        # distractors
        resource_files=None,
        sensor_noise_mult=0,
        spatial_jitter=0,
        # others
        camera_id=0,
        environment_kwargs=None,
        task_kwargs={},
        visualize_reward=False,
    )

    # Sec. B.1.1
    if domain_name == 'walker' and task_name == 'walk':
        background_remove_mode = 'argmax'
    else:
        background_remove_mode = 'dbc'

    if variant == 'noiseless':
        pass
    elif variant == 'video_background':
        kwargs.update(
            resource_files=os.path.expanduser("~/.driving_car/*.mp4"),
            background_remove_mode=background_remove_mode,
        )
    elif variant == 'video_background_noisy_sensor':
        kwargs.update(
            resource_files=os.path.expanduser("~/.driving_car/*.mp4"),
            sensor_noise_mult=1,
            background_remove_mode=background_remove_mode,
        )
        if f"{domain_name}_{task_name}" not in ['cheetah_run', 'walker_walk', 'reacher_easy']:
            raise RuntimeError(f'Noisy sensor not implemented for {domain_name}_{task_name}')
    elif variant == 'video_background_camera_jitter':
        kwargs.update(
            resource_files=os.path.expanduser("~/.driving_car/*.mp4"),
            spatial_jitter=120,
            background_remove_mode=background_remove_mode,
        )
    else:
        raise ValueError(f"Unexpected environment")

    return DMCWrapper(seed=seed, **kwargs)
