from xhand_controller import xhand_control as xh
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# =========================
# Model & Device Config
# =========================

JIT_POLICY_PATH = "./checkpoints/screwdriver.pt"
SCREWDRIVER = True
DEVICE = torch.device("cpu")

jit_model = torch.jit.load(JIT_POLICY_PATH, map_location=DEVICE)
jit_model.eval()
print(f"Successfully loaded JIT policy from {JIT_POLICY_PATH}")

ACTION_SCALE = 0.04167

# =========================
# Index Mapping
# =========================

XHAND_TO_POLICY_IDX = [3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2]
POLICY_TO_XHAND_IDX = [0] * len(XHAND_TO_POLICY_IDX)
for i, idx in enumerate(XHAND_TO_POLICY_IDX):
    POLICY_TO_XHAND_IDX[idx] = i


def convert_xhand_to_policy_order(vec):
    """Reorder a 12-dof vector from xHand order to policy order."""
    vec = np.asarray(vec)
    return vec[XHAND_TO_POLICY_IDX]


def convert_policy_to_xhand_order(vec):
    """Reorder a 12-dof vector from policy order to xHand order."""
    vec = np.asarray(vec)
    return vec[POLICY_TO_XHAND_IDX]


# =========================
# Observation / Proprio Config
# =========================

OBS_DIM = 96
PROPRIO_HIST_LEN = 30
PROPRIO_DIM = 24
CLIP_THUMB_RANGE = True

# Proprio history buffer
base = torch.tensor(
    [
        -0.17, 1.3, 0.44, 1.3, 0.44, 0.0, 0.0, 0.0,
        0.0, 1.3, 0.5, 0.45, -0.17, 1.3, 0.44, 1.3,
        0.44, 0.0, 0.0, 0.0, 0.0, 1.3, 0.5, 0.45,
    ],
    device=DEVICE,
    dtype=torch.float32,
)
proprio_hist_buffer = base.view(1, 1, PROPRIO_DIM).repeat(1, PROPRIO_HIST_LEN, 1)

# Joint limits
xhand_dof_lower_limits = np.array(
    [-0.1750, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, -1.0500, -0.1700],
    dtype=np.double,
)

xhand_dof_upper_limits = np.array(
    [0.1750, 1.9200, 1.9200, 1.9200, 1.9200, 1.9200,
     1.9200, 1.9200, 1.9200, 1.8300, 1.5700, 1.8300],
    dtype=np.double,
)

if CLIP_THUMB_RANGE:
    xhand_dof_lower_limits[9] = 0.6
    xhand_dof_upper_limits[9] = 1.73

# =========================
# Normalization stats from model
# =========================

obs_mean = jit_model.running_mean.unsqueeze(0)
obs_var = jit_model.running_var.unsqueeze(0)
obs_count = jit_model.running_count

sa_mean = jit_model.sa_mean.unsqueeze(0)
sa_var = jit_model.sa_var.unsqueeze(0)
sa_count = jit_model.sa_count


def update_proprio_hist(q, target_q):
    """Update global proprioceptive history buffer."""
    global proprio_hist_buffer

    current_dof = torch.from_numpy(q).float().to(DEVICE)
    target_dof = torch.from_numpy(target_q).float().to(DEVICE)

    current_proprio = torch.cat([current_dof, target_dof], dim=0).unsqueeze(0).unsqueeze(0)
    prev_obs_buf = proprio_hist_buffer[:, 1:].clone()
    proprio_hist_buffer.copy_(torch.cat([prev_obs_buf, current_proprio], dim=1))


class XHandControl:
    """Minimal wrapper for xHand hardware control used in this script."""

    def __init__(self, hand_id=0, position=0.1, mode=3):
        self._hand_id = hand_id
        self._device = xh.XHandControl()
        self._hand_command = xh.HandCommand_t()

        for i in range(12):
            finger_cmd = self._hand_command.finger_command[i]
            finger_cmd.id = i
            finger_cmd.kp = 100
            finger_cmd.ki = 0
            finger_cmd.kd = 1
            finger_cmd.position = position
            finger_cmd.tor_max = 100
            finger_cmd.mode = mode

    # -------- Device management --------

    def _enumerate_devices(self, protocol: str):
        """Enumerate device hardware input ports for given protocol."""
        serial_port = self._device.enumerate_devices(protocol)
        print(f"=@= xhand devices port ({protocol}): {serial_port}")
        return serial_port

    def open_device(self, device_identifier: dict):
        """Open the hand device with given configuration.

        device_identifier:
            {
                "protocol": "RS485" or "EtherCAT",
                "serial_port": "...",      # RS485 only
                "baud_rate": 115200,       # RS485 only
            }
        """
        print("//================================")
        print("// Open hand device")
        print("//================================")

        rsp = None
        protocol = device_identifier.get("protocol")

        if protocol == "RS485":
            device_identifier["baud_rate"] = int(device_identifier["baud_rate"])
            rsp = self._device.open_serial(
                device_identifier["serial_port"],
                device_identifier["baud_rate"],
            )
            print(f"=@= open RS485 result: {rsp.error_code == 0}")

        elif protocol == "EtherCAT":
            ether_cat = self._enumerate_devices("EtherCAT")
            print(f"enumerate_devices_ethercat ether_cat = {ether_cat}")
            if ether_cat is None or not ether_cat:
                print("enumerate_devices_ethercat get empty")
            rsp = self._device.open_ethercat(ether_cat[0])
            print(f"=@= open EtherCAT result: {rsp.error_code == 0}")
        else:
            print(f"Unsupported protocol: {protocol}")
            return False

        if rsp is None or rsp.error_code != 0:
            error_msg = getattr(rsp, "error_message", "Unknown error")
            print(
                f"=@= open device error: {error_msg}. "
                f"Please check serial_port and connection"
            )
            return False

        return True

    # -------- Command & State --------

    def send_command(self):
        """Send current hand command buffer to device."""
        _ = self._device.send_command(self._hand_id, self._hand_command)

    def read_joint_pos(self, finger_id=2, force_update=True):
        """Read joint position of given finger."""
        error_struct, state = self._device.read_state(self._hand_id, force_update)
        if error_struct.error_code != 0:
            print(f"=@= xhand read_state error: {error_struct.error_message}")
            return
        return state.finger_state[finger_id].position


# =========================
# Main control loop
# =========================

def main():
    # Default hand ID is 0, finger position is 0.1, mode is 3
    controller = XHandControl(hand_id=0, position=0.1, mode=3)

    device_identifier = {"protocol": "EtherCAT"}
    if not controller.open_device(device_identifier):
        sys.exit(1)

    if SCREWDRIVER:
		prev_target_policy = np.array(
			[-0.17, 1.2, 0.4, 1.2, 0.4, 0, 0, 0, 0, 1.15, 0.6, 0.1],
			dtype=np.double,
		)  # NOTE: screwdriver init
	else:
		prev_target_policy = np.array(
			[-0.17, 1.1, 0.4, 1.1, 0.4, 0, 0, 0, 0, 1.3, 0.6, 0.45],
			dtype=np.double,
		)  # NOTE: nut bolt init


    prev_target_xhand = convert_policy_to_xhand_order(prev_target_policy)
    for i in range(12):
        controller._hand_command.finger_command[i].position = prev_target_xhand[i]

    controller.send_command()
    time.sleep(1)

    step = 0
    prev_target = prev_target_policy.copy()
    obs = torch.zeros(1, OBS_DIM, device=DEVICE, dtype=torch.float32)

    while True:
        q_xhand = []
        for i in range(12):
            pos = controller.read_joint_pos(i)
            q_xhand.append(pos)

        q_policy = convert_xhand_to_policy_order(q_xhand)

        if step % 10 == 0:
            with torch.no_grad():
                obs = torch.clamp(obs, -5.0, 5.0)
                obs = F.pad(obs, (0, OBS_DIM - obs.shape[1]))
                norm_obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-5)
                norm_obs = torch.clamp(norm_obs, -5.0, 5.0)

                norm_proprio_hist = (proprio_hist_buffer - sa_mean) / torch.sqrt(sa_var + 1e-5)
                norm_proprio_hist = torch.clamp(norm_proprio_hist, -5.0, 5.0)

                point_cloud_info = torch.zeros((1, 100, 3))

                input_dict = {
                    "obs": norm_obs,
                    "proprio_hist": norm_proprio_hist,
                    "point_cloud_info": point_cloud_info,
                }

                mu, extrin, extrin_gt = jit_model(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
                print("action: ", mu)

                jit_action = mu.cpu().numpy().flatten()
                if SCREWDRIVER:
                    jit_action[5:7] = 0
                else:
                    jit_action[5:9] = 0

            target_q = jit_action * ACTION_SCALE + prev_target
            target_q = np.clip(
                target_q,
                xhand_dof_lower_limits,
                xhand_dof_upper_limits,
            )
            prev_target = target_q.copy()

            obs = proprio_hist_buffer[:, -3:, :].reshape(1, -1).clone()
            update_proprio_hist(q_policy, prev_target)

        target_q_xhand = convert_policy_to_xhand_order(target_q)
        for i in range(12):
            controller._hand_command.finger_command[i].position = target_q_xhand[i]

        controller.send_command()
        time.sleep(0.005)
        step += 1


if __name__ == "__main__":
    main()

