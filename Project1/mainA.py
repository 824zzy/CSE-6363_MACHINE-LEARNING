from misc import *

x_train, y_train = genDummy(0, 1, 10, 0.1)
x_test = np.linspace(0, 1, 1000)
y_test = np.sin(2*np.pi*x_test)
plot(x_train, y_train, x_test, y_test)
