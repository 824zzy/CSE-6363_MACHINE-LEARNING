from misc import *

x_train, y_train = genDummy(0, 1, 10, 0.1)
x_test = np.linspace(0, 1, 1000)
y_test = np.sin(2*np.pi*x_test)

for i, degree in enumerate([1, 2, 3, 9, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
# for i, degree in enumerate([0, 1, 3, 9]):
    plt.subplot(4, 4, i+1)
    feature = FeatureFormation(degree)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y = model.predict(X_test)
    
    plt.title("M: "+str(degree))
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)
    plt.annotate("M={}".format(degree), xy=(-0.15, 1))
plt.show()