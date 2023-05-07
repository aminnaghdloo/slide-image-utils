import sys
import math

if __name__ == '__main__':
    xc = float(sys.argv[1])
    yc = float(sys.argv[2])
    r = float(sys.argv[3])
    theta_shift = float(sys.argv[4])
    for theta in range(0, 360, 120):
        x = xc + r * math.sin(math.radians(theta + theta_shift))
        y = yc + r * math.cos(math.radians(theta + theta_shift))

        print(f"0 {round(x,1)} {round(y,1)} -100.0")
