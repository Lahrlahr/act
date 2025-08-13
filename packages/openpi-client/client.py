from openpi_client import image_tools
from openpi_client import websocket_client_policy

from sim_env import BOX_POSE
from utils import sample_box_pose, sample_insertion_pose # robot functions

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=2231)

# if 'sim_transfer_cube' in task_name:
#     BOX_POSE[0] = sample_box_pose()  # used in sim reset
# elif 'sim_insertion' in task_name:
#     BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
BOX_POSE[0] = sample_box_pose()
from sim_env import make_sim_env
env = make_sim_env('sim_transfer_cube_scripted')
ts = env.reset()

for step in range(10):
    obs = ts.observation
    dict = client.infer(obs)
    ts = env.step(dict["actions"][0])
