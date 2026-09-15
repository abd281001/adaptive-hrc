from src.adaptive_agent import AdaptiveAgent
from src.models import Settings
from src.real_robot.config import load_lab_config
from src.real_robot.domain import PhysicalObservation
from src.real_robot.freeform_domain import FreeformPhysicalDomain


CONFIG = "src/real_robot/robot_configs/stretch3_three_box_hrc.json"


def domain():
    config = load_lab_config(CONFIG)
    value = FreeformPhysicalDomain(config)

    value.set_initial_scene({
        "S0_left": {
            "status": "occupied",
            "tag_id": 1,
        },
        "S0_center": {
            "status": "occupied",
            "tag_id": 5,
        },
        "S0_right": {
            "status": "occupied",
            "tag_id": 3,
        },
        "S2": {
            "status": "empty",
            "tag_id": None,
        },
        "S3": {
            "status": "empty",
            "tag_id": None,
        },
        "S4": {
            "status": "empty",
            "tag_id": None,
        },
    })

    return value


def test_freeform_action_can_repeat_after_box_moves_away():
    d = domain()
    state = d.initial_state()

    state = d.replay_transition(
        state,
        "MOVE_B5_TO_S4",
    )
    state = d.replay_transition(
        state,
        "MOVE_B5_TO_S2",
    )
    state = d.replay_transition(
        state,
        "MOVE_B5_TO_S4",
    )

    assert d.describe_state(state)["b5"] == "S4"


def test_occupied_location_is_masked():
    d = domain()
    state = d.initial_state()

    state = d.replay_transition(
        state,
        "MOVE_B1_TO_S2",
    )

    assert (
        d.successor(
            state,
            "MOVE_B5_TO_S2",
        )
        is None
    )


def test_agent_accepts_demo_longer_than_three_actions():
    d = domain()

    agent = AdaptiveAgent(
        Settings(
            seed=7,
            verbose=False,
            irl_cold_steps=1,
            irl_warm_steps=1,
            irl_horizon=12,
        ),
        domain=d,
    )

    sequence = [
        "MOVE_B1_TO_S2",
        "MOVE_B5_TO_S3",
        "MOVE_B3_TO_S4",
        "MOVE_B1_TO_S0_LEFT",
        "MOVE_B5_TO_S2",
        "MOVE_B3_TO_S3",
    ]

    state = d.initial_state()
    agent.start_demo()

    for token in sequence:
        after = d.replay_transition(
            state,
            token,
        )
        agent.observe(
            PhysicalObservation(
                state,
                token,
                after,
            )
        )
        state = after

    match = agent.end_demo()

    assert agent.demo_counter == 1
    assert match.recipe_id is not None
