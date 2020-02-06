from misc import *

x_train, y_train = genDummy(0, 1, 10, 0.1)
x_test = np.linspace(0, 1, 1000)
y_test = np.sin(2*np.pi*x_test)
# plot(X_train, y_train, X_test, y_test)