# need to install kociemba
import control
import cube_status
import cv2
import kociemba

cube = cube_status.CubeStatus()
controler = control.RubicControler()
controler.prepare()
current_status = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
side_to_color = {'U': 'O', 'R': 'B', 'F': 'W', 'D': 'R', 'L': 'G', 'B': 'Y'}  # default side_to_color

def detect():
    global current_status, side_to_color
    current_status, side_to_color = cube.detect_status()

def scramble():
    global current_status, side_to_color
    num_moves = int(input("How many random moves you want to perform?\n"))
    moves = controler.random_move(num_moves)
    print(moves)
    current_status = cube.change_status(current_status, moves)

def solve():
    global current_status, side_to_color
    moves = kociemba.solve(current_status).split()
    print(moves)
    for move in moves:
        controler.turn(move)
    current_status = cube.change_status(current_status, moves)

def customized_moves():
    global current_status, side_to_color
    command = input('Input command, enter to exit!\n')
    moves = command.split()
    for move in moves:
        controler.turn(move)
    current_status = cube.change_status(current_status, moves)

def test_motors():
    print('Testing motors.')
    moves = ["L", "L'", "R", "R'", "U", "U'", "B", "B'", "D", "D'", "F", "F'"]
    for move in moves:
        controler.turn(move)

def main():
    global current_status, side_to_color

    while True:
        img = cube.display_status(list(current_status), side_to_color)
        cv2.imshow('Current Status', img)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break

        choice = input("Choose the action you would like to perform, enter to exit. If your cube is not solved, you should perform status detection first.\n"
                       "1. Detect status\n"
                       "2. Scramble\n"
                       "3. Solve\n"
                       "4. Perform specific moves\n"
                       "5. Test motors\n")

        if choice == '1':
            detect()
        elif choice == '2':
            scramble()
        elif choice == '3':
            solve()
        elif choice == '4':
            customized_moves()
        elif choice == '5':
            test_motors()
        elif choice == '':
            break
        else:
            print("Choice not supported. Please choose again.\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
