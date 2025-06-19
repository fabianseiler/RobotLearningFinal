import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np
import tf
from pick_from_vision_V4 import ROS_OM_Node
import json
from LLM_TOH.toh_llm import main_llm, SpaceModel

class TOH:
    def __init__(self, joint_names, request_topic='/object_request', rate_hz=1,tower_a=[],tower_b=[],tower_c=[]):
        rospy.init_node('continuous_pick_and_place', anonymous=True)
        # Publisher to request object detection
        self.request_pub = rospy.Publisher(request_topic, String, queue_size=1)
        self.listener = tf.TransformListener()
        # Instantiate the motion executor
        self.om_node = ROS_OM_Node(joint_names)
        self.rate = rospy.Rate(rate_hz)

        self.tower_origins = {
        'A': np.array([0.14, -0.14, 0.05]),
        'B': np.array([0.18, 0.0, 0.05]),
        'C': np.array([0.14, 0.14, 0.05])
        }
        self.towers = {
            'A': tower_a,
            'B': tower_b,
            'C': tower_c
        }
        self.block_height = 0.07

    def get_object_pose_world(self,target_frame="world", object_frame="detected_object"):
            try:
                self.listener.waitForTransform(target_frame, object_frame, rospy.Time(0), rospy.Duration(15.0))
                (trans, rot) = self.listener.lookupTransform(target_frame, object_frame, rospy.Time(0))

                rospy.loginfo("Object position in %s frame:", target_frame)
                rospy.loginfo("  Translation: x=%.3f, y=%.3f, z=%.3f", *trans)
                rospy.loginfo("  Orientation (quaternion): x=%.3f, y=%.3f, z=%.3f, w=%.3f", *rot)

                return np.array(trans), np.array(rot)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr("Transform failed: %s", str(e))
                return None, None

    def get_object_trans(self,obj_name):
        # Publish detection request
        self.request_pub.publish(obj_name)
        rospy.loginfo(f"Requested detection for object: '{obj_name}'")

        # Wait for tf to be available
        trans, rot = None, None
        while not rospy.is_shutdown():
            trans, rot = self.get_object_pose_world(object_frame=obj_name)
           # trans, rot = self.get_object_pose_world(object_frame=obj_name)
            if trans is not None:
                return trans
            rospy.loginfo(f"Waiting for transform for '{obj_name}'...")   # TODO: Hier zeitlichen Abbruch einbauen
            rospy.sleep(0.5)


    def execute_pick_and_place(self,pick_cube,placetower):
        trans = self.get_object_trans(pick_cube)
        if trans is None:
            rospy.logerr(f"Could not get position for {pick_cube}")
            return None
        pick_position = np.array(trans)
        pick_position = pick_position + np.array([0.0, 0.0, 0.01]) # set offset
        self.om_node.execute_pick(pick_position,
                                      np.deg2rad(40),
                                      cube_name=pick_cube
                                     )

        # Small delay before next iteration
        self.rate.sleep()

        # Get goal location
        tower_stack = self.towers.get(placetower)
        if tower_stack is None:
            rospy.logerr(f"Unknown tower '{placetower}'")
            return None

        if len(tower_stack) == 0:
            print("New Tower")
            # Turm leer, Position des Turm-Origins nehmen
            place_pos = self.tower_origins[placetower].copy()
            place_pos = place_pos + np.array([0.0, 0.0, -0.05])
            print(place_pos)
            # Da Turm leer, Block wird direkt auf Ursprung platziert
            below_block = None  # Kein Block darunter
        else:
            print("On Block")
            # Höchster Block auf dem Zielturm
            top_block = tower_stack[-1]
            top_block_pos = self.get_object_trans(top_block)
            if top_block_pos is None:
                rospy.logerr(f"Could not get position for top block '{top_block}' on tower '{placetower}'")
                return None
            place_pos = np.array(top_block_pos)
            # Stapelhöhe draufsetzen (z.B. 0.07m)
            place_pos[2] += self.block_height
            below_block = top_block


        # 3. Ausführen Place
        self.om_node.execute_place(place_pos, np.deg2rad(40), cube_name=pick_cube)
        self.rate.sleep()

            # 4. Turm-Konfiguration updaten:
            #   - pick_cube aus altem Turm entfernen
            #   - pick_cube zum neuen Turm hinzufügen
        for tower_name, blocks in self.towers.items():
            if pick_cube in blocks:
                blocks.remove(pick_cube)
                break
        tower_stack.append(pick_cube)

        rospy.loginfo(f"Moved {pick_cube} to tower {placetower} on top of {below_block}")

        # 5. Rückgabe: Block unter dem neuen (oder None, wenn leer)
        return below_block

    def get_tower_configuration(self):
        """
        Returns the current configuration of the towers.

        Returns:
         --------
        dict
            A dictionary representing the current distribution of blocks on the towers.
            Example: {'A': ['blue_cube'], 'B': ['green_cube', 'red_cube'], 'C': []}
         """
        rospy.loginfo("Returning current tower configuration.")
        # Return a copy of the dictionary to avoid unwanted external modifications
        return {tower_name: list(blocks) for tower_name, blocks in self.towers.items()}


    

    def get_all_block_positions_json(self):
        """
        Holt die x, y, z Koordinaten aller Blöcke aus den aktuellen Türmen
        und gibt sie als JSON-String zurück.
        """
        all_positions = {}

        conv = {"green_cube": "3",
                "red_cube": "2",
                "blue_cube": "1"}

        for tower_name, blocks in self.towers.items():
            for block_name in blocks:
                position = self.get_object_trans(block_name)
                if position is not None:
                    all_positions[conv[block_name]] = {
                        'x': float(position[0]),
                        'y': float(position[1]),
                        'z': float(position[2]),
                    }
                else:
                    rospy.logwarn(f"Position für Block '{block_name}' konnte nicht ermittelt werden.")

        json_result = json.dumps(all_positions, indent=4)
        print("\n📦 Blockpositionen im JSON-Format:\n")
        print(json_result)
        return json_result


def process_hanoi_commands(commands, toh):
    block_map = {
        '1': 'green_cube',
        '2': 'red_cube',
        '3': 'blue_cube'
    }

    for cmd in commands:
        if not cmd.startswith("MD") or len(cmd) != 5:
            rospy.logwarn(f"Ungültiger Befehl übersprungen: {cmd}")
            continue

        block_id = cmd[2]
        from_tower = cmd[3]
        to_tower = cmd[4]

        if block_id not in block_map:
            rospy.logwarn(f"Unbekannter Block {block_id} in Befehl {cmd}")
            continue

        cube_name = block_map[block_id]
        towers = toh.get_tower_configuration()

        # Prüfen, ob Block oben auf dem angegebenen "from_tower" liegt
        if towers[from_tower] and towers[from_tower][-1] == cube_name:
            rospy.loginfo(f"Ausführen von Befehl: {cmd} ({cube_name} von {from_tower} nach {to_tower})")
            toh.execute_pick_and_place(cube_name, to_tower)
        else:
            rospy.logwarn(f"Block {cube_name} ist nicht oben auf Turm {from_tower}, Befehl {cmd} übersprungen.")





if __name__ == '__main__':
    try:
        joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        toh = TOH(joint_names=joint_names,
                  tower_a=[], 
                  tower_b=[],
                  tower_c=[])
        
        tower_origins = {
        'A': np.array([0.14, -0.14, 0.05]),
        'B': np.array([0.18, 0.0, 0.05]),
        'C': np.array([0.14, 0.14, 0.05])
        }
        
        # --> Hier Block xyz
        toh.get_all_block_positions_json()

        # Set fields in JSON file
        labels = ["x", "y", "z"]
        converted = {outer_key: dict(zip(labels, values)) 
                     for outer_key, values in tower_origins.items()}

        with open("./LLM_TOH/States_positions.json", "r") as f:
            data = json.read(f)
        data["fields"] = converted
        with open("./LLM_TOH/States_positions.json", "w") as f:
            json.dump(data, f, indent=2)

        # Get init state of cube
        Space = SpaceModel()
        states = Space.write_states()   

        with open("./LLM_TOH/States_positions.json", "r") as f:
            data = json.read(f)
            toh.towers = data["states"]

        # --> tower config --> json
        main_llm()
        with open("./LLM_TOH/States_positions_solution.json", "r") as f:
            output_states = json.read(f)
        command_list = output_states["moves"]

        # Change command list to fit Simulation
        #from llm_interface_stuff import convert_move
        #command_list = [convert_move(move) for move in command_list]

        # Beispiel-Kommandoliste
        # green 1
        # red   2
        # blue  3
        #command_list = ["MD3CB"] 

        process_hanoi_commands(command_list, toh)

        # Zeige Endzustand
        print(toh.get_tower_configuration())

    except rospy.ROSInterruptException:
        pass

