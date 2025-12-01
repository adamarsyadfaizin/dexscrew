We provide scripts for baselines with different sensing capabilities and privileged info configurations.

### Teacher policy: toggle privileged info
You can enable/disable different inputs to the Teacher for ablations, similar to the penspin repo.

- Without tactile information:
```bash
scripts/screwdriver_teacher.sh train.ppo.enable_tactile=False
```

- Without point cloud information:
```bash
scripts/screwdriver_teacher.sh task.env.hora.point_cloud_sampled_dim=0 train.ppo.use_point_cloud_info=False
```

- Without privileged information to the policy (no priv vector fed to the actor/critic):
```bash
scripts/screwdriver_teacher.sh train.ppo.priv_info=False
```

- With few point cloud points (example: 100 points):
```bash
scripts/screwdriver_teacher.sh task.env.hora.point_cloud_sampled_dim=100 train.ppo.use_point_cloud_info=True
```

- With all signals on (privileged info + point cloud):
```bash
scripts/screwdriver_teacher.sh train.ppo.priv_info=True train.ppo.use_point_cloud_info=True
```

### Important: Student-Teacher consistency (dimension matching)
- The Student policy (padapt) and the Teacher policy must use the SAME privileged info and input settings, otherwise model input dimensions will not match.
- Concretely, keep the following flags consistent between Teacher training and Student training:
  - `train.ppo.priv_info`
  - `train.ppo.use_point_cloud_info`
  - `task.env.hora.point_cloud_sampled_dim` (if using point cloud)

Example pair (Teacher then Student) with matching settings:
```bash
# Teacher
scripts/screwdriver_teacher.sh \
  train.ppo.priv_info=True \
  train.ppo.use_point_cloud_info=True \
  task.env.hora.point_cloud_sampled_dim=100

# Student (use the checkpoint produced by the Teacher above)
scripts/screwdriver_student_padapt.sh \
  train.ppo.priv_info=True \
  train.ppo.use_point_cloud_info=True \
  task.env.hora.point_cloud_sampled_dim=100
```