class BackgroundLoader:
    def __init__(self, DatabasePath: str, debug: bool = False):
        self.DatabasePath = DatabasePath
        self.debug = debug