# Wrappers

Modified from ikostrikov/jaxrl


## metaworld

['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

DecQN benchmarks on
* door-open-v2
* hammer-v2
* pick-place-v2
* assembly-v2
* drawer-open-v2

run with `metaworld_{env-name}`


## robosuite

Environments are created `{Robot}_{controller}-{impedance}_{name}`

* Lift
* Stack
* NutAssembly
* NutAssemblySingle
* NutAssemblySquare
* NutAssemblyRound
* PickPlace
* PickPlaceSingle
* PickPlaceMilk
* PickPlaceBread
* PickPlaceCereal
* PickPlaceCan
* Door
* Wipe

These environments require two robots and have to be created `{Robot}-{Robot}_` or `Baxter+` (as Baxter has two arms)
* {Robot}-{Robot}_TwoArmLift
* {Robot}-{Robot}_TwoArmPegInHole
* {Robot}-{Robot}_TwoArmHandover

We benchmark on 
* Panda_OSC_POSE-fixed_Lift
* Panda_OSC_POSE-fixed_Stack
* Panda_OSC_POSE-fixed_PickPlaceCan
* Panda_OSC_POSE-fixed_NutAssemblyRound
* Panda_OSC_POSE-fixed_Door
* Panda_OSC_POSE-fixed_Wipe
* Panda-Panda_OSC_POSE-fixed_TwoArmLift
* Panda-Panda_OSC_POSE-fixed_TwoArmPegInHole
* Panda-Panda_OSC_POSE-fixed_TwoArmHandover


Robots are
* Panda (7)
* Sawyer (7)
* IIWA (7)
* Jaco (7)
* Kinova3 (7)
* UR5e (6)
* Baxter (7+7)