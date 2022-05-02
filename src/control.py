import random
import serial
import time

class RubicControler:
    def __init__(self):
        self.gcode_map = {
                "D": [b'G0 X-10\n'],
                "D'": [b'G0 X10\n'],
                "D2": [b'G0 X20\n'],
                "F": [b'G0 Y10\n'],
                "F'": [b'G0 Y-10\n'],
                "F2": [b'G0 Y20\n'],
                "R": [b'G0 Z-2\n'],
                "R'": [b'G0 Z2\n'],
                "R2": [b'G0 Z4\n'],
                "L": [b'T1\n', b'G0 E-1.6\n'],
                "L'": [b'T1\n', b'G0 E1.6\n'],
                "L2": [b'T1\n', b'G0 E3.2\n'],
                "B": [b'T0\n', b'G0 E1.6\n'],
                "B'": [b'T0\n', b'G0 E-1.6\n'],
                "B2": [b'T0\n', b'G0 E-3.2\n'],
                "U": [b'T2\n', b'G0 E-1.6\n'],
                "U'": [b'T2\n', b'G0 E1.6\n'],
                "U2": [b'T2\n', b'G0 E3.2\n']
        }

        self.ser = serial.Serial('/dev/tty.usbmodem1422401', 115200)

        # Set to relative position mode
        self.ser.write(b'G91\n')
        # Disable cold extrusion checking
        self.ser.write(b'M302 P1\n')
        # Slower extruder speed (for L, B and U) to 5
        self.ser.write(b'M203 E5\n')

    def turn(self, command):
        if not command in self.gcode_map.keys():
            return
        
        for gcode in self.gcode_map[command]:
            self.ser.write(gcode)

    def random_move(self, num_moves):
        moves = [ x for x in list(self.gcode_map.keys())]
        for i in range(num_moves):
            move = random.choice(moves)
            print(move)
            self.turn(move)
            time.sleep(0.5)


