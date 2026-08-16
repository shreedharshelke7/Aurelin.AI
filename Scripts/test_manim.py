from manim import *

class TestScene(Scene):
    def construct(self):
        text = Text("Aurelin AI")
        self.play(Write(text))
        self.wait(5)