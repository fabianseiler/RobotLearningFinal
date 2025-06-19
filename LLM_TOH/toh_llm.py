# simplified_toh_rag.py
import re
import json
import copy
from rag_call import load_documents, OllamaEmbeddings, RecursiveCharacterTextSplitter
from rag_call import Chroma, ChatPromptTemplate, ChatOllama, RunnablePassthrough, StrOutputParser
from toh_visual import TowerOfHanoi, parse_and_move
from matplotlib import pyplot as plt
from langchain.schema import Document

from Interface_coords_llm import SpaceModel


# Manages Tower of Hanoi game state and move validation
class TowerOfHanoiState:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = self.load_json()
        self.moves = []
        self.data["init_states"] = copy.deepcopy(self.data['states'])


    def convert_move(self, llm_cmd: str) -> str:
        """
        Converts the LLM command to an appropriate move for the simulation
        """
        colors = {"1": "Blue",
                  "2": "Red",
                  "3": "Green",
                  "4": "Orange",
                  "5": "Yellow"}
        return colors[llm_cmd[2]] + llm_cmd[3:]

    def load_json(self):
        try:
            with open(self.json_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None

    # Returns current peg configuration as formatted string
    def get_current_state_string(self):
        states = self.data.get("states", {})
        return f"Peg A: {states.get('A', [])}, Peg B: {states.get('B', [])}, Peg C: {states.get('C', [])}"

    # Validates and executes a single move (format: MDxYZ)
    def apply_move(self, move_code):
        if not re.match(r'MD\d[A-C][A-C]', move_code):
            return False, f"Invalid move format: {move_code}"

        disk = int(move_code[2])
        from_peg = move_code[3]
        to_peg = move_code[4]

        states = self.data["states"]

        # Validate move constraints
        if not states[from_peg]:
            return False, f"No disks on peg {from_peg}"

        top_disk = int(states[from_peg][-1]) if states[from_peg][-1] else 0
        if top_disk != disk:
            return False, f"Disk {disk} not on top of peg {from_peg}. Top disk is {top_disk}"

        if states[to_peg]:
            dest_top_disk = int(states[to_peg][-1])
            if dest_top_disk < disk:
                return False, f"Cannot place disk {disk} on smaller disk {dest_top_disk}"

        # Execute move
        moved_disk = states[from_peg].pop()
        states[to_peg].append(moved_disk)
        self.moves.append(move_code)

        return True, f"Moved disk {moved_disk} from peg {from_peg} to peg {to_peg}"

    # Checks if puzzle is solved (all disks on peg C in correct order)
    def is_solved(self):
        states = self.data["states"]
        c_disks = [int(x) for x in states["C"]] if states["C"] else []
        return (len(c_disks) == 3 and
                c_disks == [3, 2, 1] and
                not states["A"] and
                not states["B"])

    def save_with_moves(self, output_file=None):
        if output_file is None:
            output_file = self.json_file_path.replace('.json', '_solution.json')

        #self.data["moves"] = self.moves
        self.data["moves"] = []
        for move in self.moves:
            dmp_move = self.convert_move(move)
            self.data["moves"].append(dmp_move)
            print(f"Moving {move}: {dmp_move}")

        try:
            with open(output_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False

    def reset_to_initial_state(self):
        original_data = self.load_json()
        if original_data:
            self.data["states"] = copy.deepcopy(original_data["states"])
            self.moves = []
            print(f"Reset to initial state: {self.get_current_state_string()}")
            return True
        return False

    # Validates a sequence of moves without applying them
    def validate_move_sequence(self, moves):
        temp_states = copy.deepcopy(self.data["states"])

        for i, move_code in enumerate(moves):
            if not re.match(r'MD\d[A-C][A-C]', move_code):
                return False, f"Invalid move format at position {i}: {move_code}"

            disk = int(move_code[2])
            from_peg = move_code[3]
            to_peg = move_code[4]

            if not temp_states[from_peg]:
                return False, f"Move {i + 1} ({move_code}): No disks on peg {from_peg}"

            top_disk = int(temp_states[from_peg][-1]) if temp_states[from_peg][-1] else 0
            if top_disk != disk:
                return False, f"Move {i + 1} ({move_code}): Disk {disk} not on top of peg {from_peg}. Top disk is {top_disk}"

            if temp_states[to_peg]:
                dest_top_disk = int(temp_states[to_peg][-1])
                if dest_top_disk < disk:
                    return False, f"Move {i + 1} ({move_code}): Cannot place disk {disk} on smaller disk {dest_top_disk}"


            moved_disk = temp_states[from_peg].pop()
            temp_states[to_peg].append(moved_disk)
            # Check if solved after this move
            c_disks = [int(x) for x in temp_states["C"]] if temp_states["C"] else []
            is_solved = (len(c_disks) == 3 and
                         c_disks == [3, 2, 1] and
                         not temp_states["A"] and
                         not temp_states["B"])

            if is_solved:
                # Return only the moves needed up to this point
                needed_moves = moves[:i + 1]
                return True, f"Solved after {i + 1} moves: {needed_moves}"

        return True, "All moves are valid"


def load_documents_with_json(sources):

    all_docs = []

    for source in sources:
        docs = load_documents([source])
        if docs:
            all_docs.extend(docs)
            print(f"Loaded document: {source}")
        else:
            print(f"Error loading {source}:")

    return all_docs


# Sets up RAG chain with document retrieval and LLM for generating moves
def setup_simple_rag(sources):
    print("Setting up RAG...")

    html_sources = [s for s in sources if not s.endswith('.json')]

    # Load and chunk documents
    data = load_documents(html_sources)
    if not data:
        print("Failed to load documents")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)

    # Initialize embeddings
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings.embed_query("test")
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

    # Create vector store and retriever
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            collection_name="toh-rag",
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever()
    except Exception as e:
        print(f"Vector store error: {e}")
        return None

    # Initialize chat model
    try:
        model = ChatOllama(model='gemma3:27b')
        model.invoke("test")
    except Exception as e:
        print(f"Chat model error: {e}")
        return None

    # Define prompt template for move generation
    template = """You are solving a Tower of Hanoi puzzle. Based on the solution format in the context, provide the moves needed.

Context: {context}

Current game state: {current_state}

Rules:
- Move only one disk at a time
- Never place a larger disk on a smaller disk  
- Goal: Move all disks from peg A to peg C
- Use format MDxYZ where x=disk number, Y=source peg, Z=destination peg
- You can only move the topmost disk from each peg

If this is a retry (partial moves already made), continue from the current position.

Provide ONLY the move codes, one per line:
"""

    prompt = ChatPromptTemplate.from_template(template)

    # Build RAG chain
    chain = (
            {"context": retriever, "current_state": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    print("RAG setup complete")
    return chain


# Extracts move codes from LLM response text
def extract_moves(text):
    return re.findall(r'MD\d[A-C][A-C]', text)


# Main solving loop with retry mechanism
def solve_with_retry(chain, game_state, max_attempts=3):
    for attempt in range(max_attempts):
        print(f"\n--- Attempt {attempt + 1} ---")

        current_state = game_state.get_current_state_string()
        print(f"Current state: {current_state}")

        if attempt > 0:
            print(f"Previous moves: {', '.join(game_state.moves)}")

        try:
            # Get LLM response and extract moves
            response = chain.invoke(current_state)
            print(f"LLM response:\n{response}")

            moves = extract_moves(response)
            if not moves:
                print("No valid moves found")
                continue

            print(f"Extracted moves: {moves}")

            # Validate moves before applying
            valid, validation_msg = game_state.validate_move_sequence(moves)
            if not valid:
                print(f"Move validation failed: {validation_msg}")
                continue

            # Apply moves sequentially
            move_success = True
            for i, move in enumerate(moves):
                success, msg = game_state.apply_move(move)
                if success:
                    print(f"Move {i + 1} - {move}: {msg}")
                    print(f"  State: {game_state.get_current_state_string()}")
                else:
                    print(f"Move {i + 1} - {move}: {msg}")
                    move_success = False
                    break

            if game_state.is_solved():
                print("\nSolved")
                return True
            elif move_success:
                print("Moves applied, retrying...")
            else:
                print("Invalid moves, retrying...")

        except Exception as e:
            print(f"Error: {e}")

    print(f"Failed to solve after {max_attempts} attempts")
    return False


# Main execution flow
def main():

    # Update State positions
    Space = SpaceModel("States_positions.json",
                       x_dev=0.1, y_dev=0.1)
    Space.write_states(json_file="States_positions.json")

    print("Loading game state...")
    game_state = TowerOfHanoiState("States_positions.json")
    init_state = copy.deepcopy(game_state)
    if game_state.data is None:
        print("Failed to load JSON")
        return

    print(f"Initial state: {game_state.get_current_state_string()}")

    # Define knowledge sources for RAG
    sources = [
        "./hanoi_tagged_solution_3_disks.html",
        #"./toh.pdf" # dont include, seems to mess up the reasoning!
    ]

    chain = setup_simple_rag(sources)
    assert chain is not None, "Failed to setup RAG"

    print("--- Starting solver ---")
    success = solve_with_retry(chain, game_state, max_attempts=5)

    if success:
        print(f"Final moves: {', '.join(game_state.moves)}")
        game_state.save_with_moves()

        #viz = input("Visualize? (y/n): ")
        #if viz.lower() == 'y':
        #    visualize_solution(game_state.moves)
    else:
        print("Not solved")
        if game_state.moves:
            print(f"Partial moves: {', '.join(game_state.moves)}")
            save = input("Save partial? (y/n): ")
            if save.lower() == 'y':
                game_state.save_with_moves()


# Visualizes the solution moves using matplotlib
def visualize_solution(moves):
    ### only works for movin all disks from A to B
    print("Visualizing...")
    try:
        hanoi = TowerOfHanoi(3)
        plt.pause(1)

        for move in moves:
            if parse_and_move(hanoi, move):
                plt.pause(0.5)
            else:
                print(f"Visualization failed at move: {move}")
                break

        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")


if __name__ == "__main__":
    main()
