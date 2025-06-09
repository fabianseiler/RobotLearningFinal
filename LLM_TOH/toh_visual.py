# toh_visual.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import time


class TowerOfHanoi:
    def __init__(self, num_disks=3):
        self.num_disks = num_disks
        self.pegs = {
            'A': list(range(num_disks, 0, -1)),  # Largest to smallest
            'B': [],
            'C': []
        }
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Interactive mode
        self.draw_state()
        
    def move_disk(self, disk_num, from_peg, to_peg):
        """Move disk from one peg to another"""
        if disk_num not in self.pegs[from_peg]:
            print(f"Error: Disk {disk_num} not on peg {from_peg}")
            return False
        if self.pegs[from_peg][-1] != disk_num:
            print(f"Error: Disk {disk_num} is not on top of peg {from_peg}")
            return False
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < disk_num:
            print(f"Error: Cannot place disk {disk_num} on top of smaller disk")
            return False
        self.pegs[to_peg].append(self.pegs[from_peg].pop())
        self.draw_state()
        return True
    
    def draw_state(self):
        """Draw the current state of the Tower of Hanoi"""
        self.ax.clear()
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, self.num_disks + 1)
        self.ax.axis('off')
        peg_positions = {'A': 0.5, 'B': 1.5, 'C': 2.5}
        for peg, position in peg_positions.items():
            self.ax.add_patch(Rectangle((position-0.1, 0), 0.2, 0.2, fc='brown'))
            self.ax.add_patch(Rectangle((position-0.05, 0.2), 0.1, self.num_disks, fc='brown'))
            for i, disk in enumerate(self.pegs[peg]):
                disk_width = disk * 0.2
                self.ax.add_patch(Rectangle(
                    (position - disk_width/2, i + 0.2), 
                    disk_width, 
                    0.2, 
                    fc=plt.cm.viridis(disk/self.num_disks)
                ))
        for peg, position in peg_positions.items():
            self.ax.text(position, 0, peg, ha='center')
        self.ax.set_title('Tower of Hanoi')
        plt.pause(0.5)
        self.fig.canvas.draw()

def parse_and_move(hanoi, move_code):
    """Parse a move code like 'MD1AB' and execute it"""
    try:
        disk_num = int(move_code[2])
        from_peg = move_code[3]
        to_peg = move_code[4]
        print(f"Moving disk {disk_num} from peg {from_peg} to peg {to_peg}")
        return hanoi.move_disk(disk_num, from_peg, to_peg)
    except (IndexError, ValueError) as e:
        print(f"Invalid move code: {move_code}. Format should be 'MDxYZ' where x is disk number, Y is source peg, Z is destination peg")
        return False

# Example usage function
if __name__ == "__main__":
    hanoi = TowerOfHanoi(3)
    plt.pause(1) # Wait for user to see initial state
    
    # Corrected example moves for 3 disks from A to C
    moves = ["MD1AC", "MD2AB", "MD1CB", "MD3AC", "MD1BA", "MD2BC", "MD1AC"] 
    
    all_moves_successful = True
    for move in moves:
        if not parse_and_move(hanoi, move):
            print(f"Stopping due to invalid move: {move}")
            all_moves_successful = False
            break 
    
    print("\nExample moves execution finished.")
    if all_moves_successful and len(hanoi.pegs['C']) == hanoi.num_disks and not hanoi.pegs['A'] and not hanoi.pegs['B']:
        print("Puzzle solved correctly in the example!")
    else:
        print("Puzzle not solved correctly in the example.")
        print(f"Final peg states: A: {hanoi.pegs['A']}, B: {hanoi.pegs['B']}, C: {hanoi.pegs['C']}")

    plt.ioff()
    plt.show()