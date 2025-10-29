# Keith Briggs 2025-07-10

try:
    from os import getuid
    from pwd import getpwuid

    def get_full_username():
        "Returns the full username of the current user."
        return getpwuid(getuid())[4].strip(",")

except:
    from os import getlogin

    def get_full_username():
        "For Windows, not tested"
        return getlogin()


from time import strftime, localtime
from random import choice as random_choice
import numpy as np
import matplotlib.pyplot as plt


def red(t):
    return f"\033[91m{t}\033[0m"


def green(t):
    return f"\033[92m{t}\033[0m"


def yellow(t):
    return f"\033[33m{t}\033[0m"


def blue(t):
    return f"\033[94m{t}\033[0m"


def bold(t):
    return f"\033[1m{t}\033[0m"


def purple(t):
    return f"\033[0;35m{t}\033[0m"


def cyan(t):
    return f"\033[36m{t}\033[0m"


def bright_yellow(t):
    return f"\033[93m{t}\033[0m"


def random_negative_emoji():
    return random_choice("🥵😬😥😠😧😠😡😢😣😤😥😦😧😨😩😪😫😬😭😰😱😲😳")


def to_dB(x):
    return 10.0 * np.log10(x)


def from_dB(x):
    return np.power(10.0, x / 10.0)


def fig_timestamp(
    fig, author, fontsize=6, color="black", alpha=0.7, rotation=0, prespace=""
):
    date = strftime("%Y-%m-%d %H:%M", localtime())
    fig.text(  # position text relative to Figure
        0.01,
        0.005,
        prespace + f"{author} {date}",
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color=color,
        rotation=rotation,
        transform=fig.transFigure,
        alpha=alpha,
    )


def move_ues_Gaussian(crrm, size_tuple, choice_tuple):
    # convenience function for Gaussian UE moves
    rng = crrm.rngs[0]
    move_size = rng.choice(size_tuple, size=1)[0]
    indices = rng.choice(choice_tuple, size=move_size, replace=False).tolist()
    deltas = crrm.params.move_mean + crrm.params.move_stdev * rng.standard_normal(
        size=(move_size, 2)
    )
    crrm.move_ue_locations(indices, deltas)
