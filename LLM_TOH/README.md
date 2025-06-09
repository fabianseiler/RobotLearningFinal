# TOH solver
![Overview](Overview.png)
This directory contains the LLM that solves the game in the green part of the above overview:
# USAGE:
### Execute
    python toh_llm.py
## INPUTS
It uses the states as inputs in this format that can also be seen in the example States_positions.json
  
    "states":
    {
    "A": [],
    "B": ["3","2","1"],
    "C": []
     }
## OUTPUTS
Outputs are in desired format and can be altered in the class function 

    class TowerOfHanoiState:
        def save_with_moves(self):

Outputs will be saved in 
    
    States_positions_solution.json

Output-format includes the final state and if wished the estimated positions of the cubes can be updated with mov.py

BUT SCRIPT NEEDS ADAPTATION TO NEW OUTPUT FORMAT THIS WILL ONLY BE IMPLEMENTED IF NEEDED!:

If you want to use this, comment out these lines like shown below_

    class TowerOfHanoiState:
         def save_with_moves(self):
        ...
        self.data["moves"] = self.moves
        #self.data["moves"] = []
        #for move in self.moves:
        #    dmp_move = self.convert_move(move)
        #    self.data["moves"].append(dmp_move)
        #    print(f"Moving {move}: {dmp_move}")
        ...


Structure of JSON files:

Fields(optional): represent the position of the field location where the blocks are going to be placed

Positions(optional): represent the initial positions of the detected cubes

init_states:

states: final states after all movement

moves: moves that will be used to play the game
    
    "fields": {
    "A": {"x": -5,"y": 17,"z": 0},
    "B": {"x": 0, "y": 17,"z": 0},
    "C": {"x": 5, "y": 17,"z": 0 }
    },

    "positions": {
    "1": {"x": -5,"y": 17,"z": 4},
    "2": {"x": -5,"y": 17,"z": 2.5},
    "3": {"x": -5,"y": 17, "z": 0.5}},

    "init_states": {
    "A": [],
    "B": ["3","2","1"],
    "C": []
    },

    "states": {
    "A": [],
    "B": [],
    "C": ["3","2","1"]
     },

    "moves": [
    "BlueBC",
    "RedBA",
    "BlueCA",
    "GreenBC",
    "BlueAB",
    "RedAC",
    "BlueBC"
    ]
 