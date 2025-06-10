# Tower of Hanoi mit OpenManipulator in Gazebo

## Startanleitung
`gazebo_ros_link_attacher` muss hinzugefügt werden. Dafür den Ordner einfach unter /src/ ablegen. Danch ``catkin_make`` ausführen und ``devel/setup.bash``.
### 1. Link-Attacher starten
Zuerst muss 
```
roslaunch gazebo_ros_link_attacher test_attacher.launch
```
Dieser Befehl startet den ROS-Link-Attacher, der für das Aufnehmen und Ablegen von Objekten in Gazebo erforderlich ist.

### 2. Simulationsumgebung mit OpenManipulator starten

```
roslaunch om_position_controller position_control.launch sim:=true
```
Damit wird die Simulationsumgebung mit dem OpenManipulator-Roboter initialisiert.

### 3. Cube-Publisher starten
```
roscd om_position_controller/scripts && python3 4_rs_detect_sim.py
```
Dieser Script publisht die Positionen der Würfel über das /tf-System, sodass der Roboter die Objekte erkennen und greifen kann.

### 4. Script starten
```
cd src/execution_scripts
```
```
python3 TOH.py
```
