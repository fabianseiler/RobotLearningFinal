from collections import defaultdict
import json
import copy


class SpaceModel:
    def __init__(self, json_file, x_dev=0.1, y_dev=0.2, tolerance=0.01):

        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.dict_coords = copy.deepcopy(self.data["positions"])

        coordinates = self.data["positions"]
        for cube in coordinates:
            coordinates[cube] = list(coordinates[cube][x] for x in coordinates[cube])

        fields = self.data["fields"]

        x_range = [fields["A"]["x"] - x_dev, fields["A"]["x"] + x_dev]
        y_range = [[fields["A"]["y"] - y_dev, fields["A"]["y"] + y_dev],
                   [fields["B"]["y"] - y_dev, fields["B"]["y"] + y_dev],
                   [fields["C"]["y"] - y_dev, fields["C"]["y"] + y_dev]]

        self.coordinates = coordinates
        self.x_range = x_range  # [x_min, x_max]
        self.y_ranges = y_range  # [[y_min, y_max], [y_min, y_max], [y_min, y_max]]
        self.tolerance = tolerance  # for fuzzy comparison
        self.toh_state = self.get_toh_states()

    def _in_range(self, val, min_val, max_val):
        """Helper to check if val is within range with tolerance"""
        return (min_val - self.tolerance) <= val <= (max_val + self.tolerance)

    def get_toh_states(self):
        peg_count = len(self.y_ranges)
        pegs = [[] for _ in range(peg_count)]  # Each peg is a list of (z, label)

        for label, (x, y, z) in self.coordinates.items():
            # Check x-range
            if not self._in_range(x, self.x_range[0], self.x_range[1]):
                raise ValueError(f"Disk '{label}' has x={x} outside allowed range {self.x_range}")

            # Assign to correct peg based on y-coordinate
            assigned = False
            for i, (ymin, ymax) in enumerate(self.y_ranges):
                if self._in_range(y, ymin, ymax):
                    pegs[i].append((z, label))
                    assigned = True
                    break

            if not assigned:
                raise ValueError(f"Disk '{label}' has y={y} outside any peg y-range")

        # Validate and sort stacks
        toh_state = []
        for stack in pegs:
            sorted_stack = sorted(stack, reverse=True)  # z descending (bottom to top)

            # Validate stack order
            for i in range(len(sorted_stack) - 1):
                if sorted_stack[i][0] <= sorted_stack[i + 1][0]:
                    raise ValueError(
                        f"Invalid stack: disk '{sorted_stack[i][1]}' (z={sorted_stack[i][0]}) "
                        f"must be below disk '{sorted_stack[i + 1][1]}' (z={sorted_stack[i + 1][0]})"
                    )
            toh_state.append([label for _, label in sorted_stack])

        toh_state = [state[::-1] if len(state) >= 1 else state for state in toh_state]

        return toh_state

    def print_state(self):
        peg_names = ['A', 'B', 'C']
        state = "Current Tower of Hanoi state:\n"
        print("Current Tower of Hanoi state:")
        for i in range(3):  # Fixed to classic 3 pegs
            peg = self.toh_state[i] if i < len(self.toh_state) else []
            peg_label = peg_names[i]
            if peg:
                state += f"Peg {peg_label}: {', '.join(peg)}\n"
                print(f"Peg {peg_label}: {', '.join(peg)}")
            else:
                state += f"Disk {peg_label}: empty\n"
                print(f"Peg {peg_label}: empty")
        return state

    def write_states(self, json_file):
        toh_state = self.toh_state

        toh_state = {"A": toh_state[0],
                     "B": toh_state[1],
                     "C": toh_state[2]}
        self.data["positions"] = self.dict_coords
        self.data["states"] = toh_state
        with open(json_file, "w") as f:
            json.dump(self.data, f)


def convert_move(llm_cmd: str) -> str:
    """
    Converts the LLM command to an appropriate move for the simulation
    """
    colors = {"1": "Blue",
              "2": "Red",
              "3": "Green",
              "4": "Orange",
              "5": "Yellow"}
    return colors[llm_cmd[2]] + llm_cmd[3:]


if __name__ == "__main__":
    space = SpaceModel("States_positions.json")
    state = space.print_state()