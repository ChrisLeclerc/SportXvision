"""Class information for SoccerNet action spotting"""

LABEL_NAMES = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card"
]

NUM_CLASSES = len(LABEL_NAMES)

CLASS_WEIGHTS = {label: 1.0 for label in LABEL_NAMES}

NMS_WINDOWS = {label: 20.0 for label in LABEL_NAMES}

TOLERANCES_TIGHT = [1, 2, 3, 4, 5]
TOLERANCES_LOOSE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
TOLERANCES_EXTRA = [0.5, 6.0, 8.0]
