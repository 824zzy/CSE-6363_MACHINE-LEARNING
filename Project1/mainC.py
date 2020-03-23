from misc import *

x_train, y_train = genDummy(0, 1, 10, 0.1)
x_test = np.linspace(0, 1, 1000)
y_test = np.sin(2*np.pi*x_test)

for i, Lambda in enumerate([1e-10, 1e-3, 1e-1, 1]):
    plt.subplot(2, 2, i+1)
    feature = FeatureFormation(9)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = RidgeRegression(Lambda)
    model.fit(X_train, y_train)
    y = model.predict(X_test)
    plt.title("lambda: "+str(Lambda))
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)
    plt.annotate("M=9", xy=(-0.15, 1))
    
plt.show()