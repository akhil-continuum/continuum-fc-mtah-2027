"""Which side (0 = home, 1 = away) a Smart Tagging ``Team`` text names. A per-game fact.

Neither ``meta.json`` nor ``games.py`` records a team name, so the table below holds the values
published in DATA.md, section "Smart Tagging" (tag group ``Team``: ``Barcelona``, ``Atletic
Club``, ``N/A``), for the game id ``BAR-ATH``: the first club of the stem is the home side. A
text that is neither in the table nor a "no team" value maps to no team (no vote), and the stage
counts such instances so that a renamed export is noticed.

``check_against_tracking`` compares the table with the home and away team names of the tracking
metadata, so that a wrong orientation shows as one word (agree, disagree, unknown).

The proper home for these names is ``meta.json`` (written by the loader from the tracking
metadata); until the loader records them, this table is the only source.
"""

# game id -> {normalised Team text: side}; texts as published in DATA.md. "athletic club" is the
# usual spelling of the same club, kept so that a corrected export and the tracking metadata's team
# name are both recognised (the check below would otherwise say "unknown").
TEAM_ALIASES = {
    "BAR-ATH": {"barcelona": 0, "atletic club": 1, "athletic club": 1},
}
NO_TEAM_TEXTS = ("", "n/a", "na", "none")     # DATA.md: `Team` can be `N/A` (every throw-in)

AGREE, DISAGREE, UNKNOWN = "agree", "disagree", "unknown"


def norm(text):
    """Lower case, single spaces, no surrounding blanks; None becomes ''."""
    return " ".join(str(text).split()).casefold() if text is not None else ""


def aliases_for(game_id):
    """{normalised team text: 0 or 1} for a game id; empty when the game has no table."""
    return dict(TEAM_ALIASES.get(game_id, {}))


def _sides_named(name, aliases):
    name = norm(name)
    if not name:
        return set()
    return {side for alias, side in aliases.items() if alias in name or name in alias}


def check_against_tracking(aliases, home_name, away_name):
    """agree / disagree / unknown: do the aliases put the tracking metadata's home team at 0?

    A name matches an alias when one contains the other after normalising ("FC Barcelona" and
    "barcelona"). Unknown when neither name matches, or when the two names contradict each other.
    """
    home, away = _sides_named(home_name, aliases), _sides_named(away_name, aliases)
    verdicts = set()
    if home == {0} or away == {1}:
        verdicts.add(AGREE)
    if home == {1} or away == {0}:
        verdicts.add(DISAGREE)
    return verdicts.pop() if len(verdicts) == 1 else UNKNOWN
