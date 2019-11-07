import numpy as np
def main():
    nx, ny = (5, 3)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    print('x = \n', x)
    print('y = \n', y)

    x2, y2 = np.meshgrid(x, y)
    print('x2 = \n', x2)
    print('y2 = \n', y2)

    #تبدیل ماتریس به آرایه
    x2r = x2.ravel()
    y2r = y2.ravel()
    print('x2r = \n', x2r)
    print('y2r = \n', y2r)

    #تبدیل به زوج مرتب دو تایی
    points = np.c_[x2r, y2r] 
    print('points = \n', points)
    
main()