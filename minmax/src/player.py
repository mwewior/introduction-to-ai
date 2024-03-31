class Player:
    def __init__(self, char: str) -> None:
        if len(char) != 1:
            raise ValueError("Mark of player must be a sign (length: 1)")
        self._char = char


class Xplayer(Player):
    def __init__(self) -> None:
        super().__init__('x')


class Oplayer(Player):
    def __init__(self) -> None:
        super().__init__('o')
