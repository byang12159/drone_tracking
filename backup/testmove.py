from backup.controller import Controller

class planner:
    def __init__(self):
        x0 = 0
        goal = 10
        dt = 1
        
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)