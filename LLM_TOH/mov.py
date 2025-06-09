import json

with open('States_positions_solution.json', 'r') as file:
    data = json.load(file)


state = data['init_states']
positions = data['positions']

heights = {
    "1" : 1,
    "2" : 1.5,
    "3" : 2
}

z_offset = 0.5

movCoor = []

for mv in data['moves']:
    cube = mv[2]
    start = mv[3]
    dest = mv[4]
    print(cube)
    startpos = positions[cube]
    endpos = data['fields'][dest]
    endpos['z'] = 0.5
    
    for block in state[dest]:
        endpos['z'] += heights[block]
    print(state[start])
    positions[cube] = endpos
    state[start].remove(cube)
    state[dest].append(cube)

    print(startpos)
    print(endpos)
    print(state)

    print("\n\n")

    movCoor.append({
        "start" : startpos,
        "end" : endpos
    })

out_file = open("start_end_positions.json", "w")

json.dump(movCoor,out_file)

out_file.close()
