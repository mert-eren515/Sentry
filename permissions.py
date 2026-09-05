import tomllib
from pathlib import Path

from gestures import ALL_GESTURES

DEFAULT_PATH = Path(__file__).parent / "permissions.toml"

# A name that stands for "anyone we recognised", and a gesture that stands for
# "all of them". Both are written as * in the file.
ANY = "*"


class Permissions:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.rules = {}
        self.load()

    def load(self):
        if not self.path.exists():
            print(f"WARNING: {self.path.name} missing, every gesture will be denied")
            self.rules = {}
            return

        with open(self.path, "rb") as f:
            self.rules = tomllib.load(f).get("permissions", {})

        # A typo in the file would otherwise fail silently as a denial, which is
        # the hardest kind of bug to notice in a system that is supposed to deny.
        for who, granted in self.rules.items():
            for gesture in granted:
                if gesture != ANY and gesture not in ALL_GESTURES:
                    print(f"WARNING: unknown gesture '{gesture}' for '{who}'")

        people = ", ".join(sorted(self.rules)) or "nobody"
        print(f"permissions loaded for: {people}")

    def allowed(self, name, gesture):
        # Denies by default: an unlisted person and an unlisted gesture both
        # fall through to False. An unidentified caller is refused outright, so
        # the everyone rule can never be reached without a name behind it.
        if not name:
            return False
        for who in (name, ANY):
            granted = self.rules.get(who, [])
            if gesture in granted or ANY in granted:
                return True
        return False
