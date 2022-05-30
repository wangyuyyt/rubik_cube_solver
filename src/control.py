import cube_status
import random
import serial
import time

class RubicControler:
    def __init__(self):
        self.gcode_map = {
                "D": ['G0 X-10'],
                "D'": ['G0 X10'],
                "D2": ['G0 X20'],
                "F": ['G0 Y10'],
                "F'": ['G0 Y-10'],
                "F2": ['G0 Y20'],
                "R": ['G0 Z-2'],
                "R'": ['G0 Z2'],
                "R2": ['G0 Z4'],
                "L": ['T1', 'G0 E-1.6'],
                "L'": ['T1', 'G0 E1.6'],
                "L2": ['T1', 'G0 E3.2'],
                "B": ['T0', 'G0 E-1.6'],
                "B'": ['T0', 'G0 E1.6'],
                "B2": ['T0', 'G0 E-3.2'],
                "U": ['T2', 'G0 E-1.6'],
                "U'": ['T2', 'G0 E1.6'],
                "U2": ['T2', 'G0 E3.2']
        }

    def prepare(self):
        self.ser = serial.Serial('/dev/tty.usbmodem1422401', 115200)
        # Set to relative position mode
        self.write_gcode('G91')
        # Disable cold extrusion checking
        self.write_gcode('M302 P1')
        # Slower extruder speed (for L, B and U) to 5
        self.write_gcode('M203 E5')

    def turn(self, command):
        if not command in self.gcode_map.keys():
            print('Command not supported.')
            return
        
        for gcode in self.gcode_map[command]:
            self.write_gcode(gcode)

    def random_move(self, num_moves):
        all_moves = [ x for x in list(self.gcode_map.keys())]
        moves = []
        for i in range(num_moves):
            move = random.choice(all_moves)
            self.turn(move)
            moves.append(move)
        return moves

    def write_gcode(self, gcode):
        #print(bytes(gcode + '\n', 'UTF-8'))
        self.ser.write(bytes(gcode + '\n', 'UTF-8'))
        time.sleep(0.1)

def main():
    default_side_to_color = {'U': 'O', 'R': 'B', 'F': 'W', 'D': 'R', 'L': 'G', 'B': 'Y'}
    cube = cube_status.CubeStatus()

    status = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    controler = RubicControler()
    controler.prepare()
    command = input('Input command, enter to exit!\n')
    moves = []
    while command != '':
        if command[0] == 'r':
            try:
              num_moves = int(command[1:])
              moves = controler.random_move(num_moves)
            except:
              print('r should be followed by a number.')
        elif command[0] == 'g':
            controler.write_gcode(command[1:])
        else:
          moves = command.split()
          for move in moves:
              controler.turn(move)

        status = cube.change_status(status, moves)
        cube.display_and_validate_status(status, default_side_to_color)

        command = input('Input command, enter to exit!\n')

if __name__ == "__main__":
    main()


