# CAGE Challenge 3 - "Canaries and Whistles" (Hicks et al., AISec 2023). See README.md and LICENSE.
# Shared helpers for the Canary agents: canary-message encode/decode and action conversion.

from CybORG.Simulator.Actions import Sleep, Action
from CybORG.Simulator.Actions.ConcreteActions.ExploitActions.RetakeControl import RetakeControl
from CybORG.Simulator.Actions.ConcreteActions.RemoveOtherSessions import RemoveOtherSessions
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTraffic
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTraffic

def unpad_canary_message(canary_msg):
    """
    Retrieve the original canary, overheard, and whistle values from the padded canary message
    """
    # Convert the list of binary digits back into an integer
    canary_msg = int(''.join(map(str, canary_msg)), 2)

    # Extract each field using a bit mask and right shift
    whistle = ((canary_msg >> 10) & 0x3F) - 1  # mask with 111111 (0x3F in hexadecimal)
    overheard = ((canary_msg >> 6) & 0x03) - 1  # mask with 11 (0x03 in hexadecimal)
    canary = (canary_msg & 0x3F) - 1  # mask with 111111 (0x3F in hexadecimal)

    # Replace -1's with None where needed
    if whistle == -1: whistle = None
    if overheard == -1: overheard = None
    if canary == -1: canary = None

    return canary, overheard, whistle

def pad_canary_message(canary, overheard, whistle):
    """
    Form a canary message from the canary, overheard, and whistle bits
    Note: +1's are added to avoid 0s in the message
    """

    # These will be transmitted as 0's after 1 is added
    if canary == None: canary = -1
    if overheard == None: overheard = -1
    if whistle == None: whistle = -1

    # Finally add 1 to whistle and canary regardless so that drone_0 heartbeat
    # can be distinguished from empty messages in obs space
    canary_msg = (int(whistle)+1)<<10 | (int(overheard)+1)<<6 | (int(canary)+1)
    canary_msg = [int(i) for i in bin(canary_msg)[2:].zfill(16)]
    return canary_msg


def cyborg_action_to_int(action):
        """
        Convert CAGE-3 CybORG action to int
        """
        if type(action)==int:
            return action
        params = action.get_params()
        if type(action) == RetakeControl:
            return 0 + int(params['agent'][11:])
        elif type(action) == RemoveOtherSessions:
            return 18
        elif type(action) == BlockTraffic:
            return 19 + int(params['agent'][11:])
        elif type(action) == AllowTraffic:
            return 37 + int(params['agent'][11:])
        elif type(action) == Sleep:
            return 55
        else:
            return -1
        
def slice_obs(obs):
    """
    Slice the observations to remove the 288 bits of comms data
    """
    sliced_obs = {}
    for agent_name in obs.keys():
        sliced_obs[agent_name] = obs[agent_name][:-288]
    return sliced_obs