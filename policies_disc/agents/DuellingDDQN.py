from policies_disc.agents.DoubleDQN  import DoubleDQN
from policies_disc.networks.duelling import Duelling


class DuellingDDQN(DoubleDQN):
    def __init__(self, obs, num_actions, args):
        obs_dim         = obs.shape[0]
        self.network_fn = lambda obs: Duelling(num_actions, args.network, args.network_groups)(obs)
        self.initialize_all(obs, args)
