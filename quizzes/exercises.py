import numpy as np

def decisionTree_modelEvaluationMetrics():
    # Model Evalutation Metrics -- Testing your models

    # Import statements
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    # Import the train test split
    # http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html
    from sklearn.cross_validation import train_test_split

    # Read in the data.
    data = np.asarray(pd.read_csv('data_decisionTree_modelEvaluationMetrics.csv', header=None))
    # Assign the features to the variable X, and the labels to the variable y.
    X = data[:,0:2]
    y = data[:,2]

    # Use train test split to split your data
    # Use a test size of 25% and a random state of 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Instantiate your decision tree model
    model = DecisionTreeClassifier()

    # TODO: Fit the model to the training data.
    model.fit(X_train, y_train)

    # TODO: Make predictions on the test data
    y_pred = model.predict(X_test)

    # TODO: Calculate the accuracy and assign it to the variable acc on the test data.
    acc = accuracy_score(y_test, y_pred)


def SVMs():
    # Import statements
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    # Read the data.
    data = np.asarray(pd.read_csv('data_SVM.csv', header=None))
    # Assign the features to the variable X, and the labels to the variable y.
    X = data[:, 0:2]
    y = data[:, 2]

    # TODO: Create the model and assign it to the variable model.
    # Find the right parameters for this model to achieve 100% accuracy on the dataset.
    model = SVC(gamma=10.0, C=6.0)
    # model = SVC(gamma=27) works well too
    # model = SVC(kernel='poly', degree=4, C=0.1) # ==> different variations of poly did not change the accuracy at all
    # model = SVC(kernel='poly', degree=200, C=200.1)

    # TODO: Fit the model.
    model.fit(X, y)

    # TODO: Make predictions. Store them in the variable y_pred.
    y_pred = model.predict(X)
    # y_pred = model.predict([[0.24539,0.81725]])
    # print(y_pred)

    # TODO: Calculate the accuracy and assign it to the variable acc.
    acc = accuracy_score(y, y_pred)

def perceptronStepCourseSolution():
    def stepFunction(t):
        if t >= 0:
            return 1
        return 0

    def prediction(X, W, b):
        return stepFunction((np.matmul(X, W) + b)[0])

    def perceptronStep(X, y, W, b, learn_rate=0.01):
        for i in range(len(X)):
            y_hat = prediction(X[i], W, b)
            if y[i] - y_hat == 1:
                W[0] += X[i][0] * learn_rate
                W[1] += X[i][1] * learn_rate
                b += learn_rate
            elif y[i] - y_hat == -1:
                W[0] -= X[i][0] * learn_rate
                W[1] -= X[i][1] * learn_rate
                b -= learn_rate
        return W, b

# this function wont work like in the course because there is a diagram function that is not shown
def perceptron_alg():
    import numpy as np
    # Setting the random seed, feel free to change it and see different solutions.
    np.random.seed(42)

    def stepFunction(t):
        if t >= 0:
            return 1
        return 0

    def prediction(X, W, b):
        return stepFunction((np.matmul(X, W) + b)[0])

    # TODO: Fill in the code below to implement the perceptron trick.
    # The function should receive as inputs the data X, the labels y,
    # the weights W (as an array), and the bias b,
    # update the weights and bias W, b, according to the perceptron algorithm,
    # and return W and b.
    def perceptronStep(X, y, W, b, learn_rate=0.01):
        # Fill in code
        n = len(y)
        # print("len y: ", len(y))
        # print("W: ", W)

        for i in range(0, n - 1):
            pred = prediction(X[i], W, b)  # X[i] is x1,x2 array, W is w1, w2 array
            # print("pred: {} -- y[{}]: {}".format(pred, i, y[i]))
            if (y[i] != pred):
                len_Xi = len(X[i])
                # print(len_Xi)
                # print(X[0])
                # print(X[1])
                if pred == 0:  # add, supposed to be 1
                    for j in range(0, len_Xi):
                        W[j] = W[j] + learn_rate * X[i][j]
                    b = b + learn_rate
                if pred == 1:  # subtract, supposed to be 0
                    for j in range(0, len_Xi):
                        W[j] = W[j] - learn_rate * X[i][j]
                    b = b - learn_rate
        return W, b

    # This function runs the perceptron algorithm repeatedly on the dataset,
    # and returns a few of the boundary lines obtained in the iterations,
    # for plotting purposes.
    # Feel free to play with the learning rate and the num_epochs,
    # and see your results plotted below.
    def trainPerceptronAlgorithm(X, y, learn_rate=0.001, num_epochs=250):
        x_min, x_max = min(X.T[0]), max(X.T[0])
        y_min, y_max = min(X.T[1]), max(X.T[1])
        W = np.array(np.random.rand(2, 1))
        b = np.random.rand(1)[0] + x_max
        # These are the solution lines that get plotted below.
        boundary_lines = []
        for i in range(num_epochs):
            # In each epoch, we apply the perceptron step.
            W, b = perceptronStep(X, y, W, b, learn_rate)
            boundary_lines.append((-W[0] / W[1], -b / W[1]))
        return boundary_lines


def regularization():
    # TODO: Add import statements
    import pandas as pd
    from sklearn.linear_model import Lasso, LinearRegression

    # Assign the data to predictor and outcome variables
    # TODO: Load the data
    train_data = pd.read_csv("data_regularization.csv", header=None)
    X = train_data.iloc[:, 0:len(train_data.columns) - 1].values
    y = train_data.iloc[:, 6:7].values

    # TODO: Create the linear regression model with lasso regularization.
    lasso_reg = Lasso().fit(X, y)

    # TODO: Fit a normal linear regression model.
    lr = LinearRegression().fit(X, y)

    # TODO: Retrieve and print out the coefficients from the regression models.
    reg_coef = lasso_reg.coef_
    print(reg_coef)

    lr_coef = lr.coef_
    print(lr_coef)

def poly_regression():
    # TODO: Add import statements
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # Assign the data to predictor and outcome variables
    # TODO: Load the data
    train_data = pd.read_csv("data.csv")
    X = train_data[["Var_X"]].values
    y = train_data["Var_Y"].values

    # Create polynomial features
    # TODO: Create a PolynomialFeatures object, then fit and transform the
    # predictor feature
    poly_feat = PolynomialFeatures(4)
    X_poly = poly_feat.fit_transform(X, y)

    # Make and fit the polynomial regression model
    # TODO: Create a LinearRegression object and fit it to the polynomial predictor
    # features
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Once you've completed all of the steps, select Test Run to see your model
    # predictions against the data, or select Submit Answer to check if the degree
    # of the polynomial features is the same as ours!

def multiple_linear_regression_in_sklearn():
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import load_boston

    # Load the data from the boston house-prices dataset
    boston_data = load_boston()
    x = boston_data['data']  # 506 x 13 features
    y = boston_data['target']  # 506 x 1

    # Make and fit the linear regression model
    # TODO: Fit the model and assign it to the model variable
    model = LinearRegression()
    model.fit(x, y) # or y.reshape(-1,1) - docs say array of shape (n_samples,) or (n_samples,n_targets)

    # Make a prediction using the model
    sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                     6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                     1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
    # TODO: Predict housing price for the sample_house
    prediction = model.predict(sample_house)
    print(prediction)

def linear_regression_in_scikit_learn():
    # TODO: Add import statements
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Assign the dataframe to this variable.
    # TODO: Load the data
    bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

    # Make and fit the linear regression model
    # TODO: Fit the model and Assign it to bmi_life_model
    bmi_life_model = LinearRegression()
    bmi_data = bmi_life_data["BMI"].reshape(-1, 1)
    le_data = bmi_life_data["Life expectancy"].reshape(-1, 1)

    # the fit function needs the X data in form (n_samples, n_features). reshape was called on the data because
    # bmi_life_data["BMI"].values shows the data is a row vector
    bmi_life_model.fit(bmi_data, le_data)

    # EASIER!! dunno why this works though
    # bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

    # Make a prediction using the model
    # TODO: Predict life expectancy for a BMI value of 21.07931
    predict = bmi_life_model.predict([21.07931])
    laos_life_exp = predict
    print(laos_life_exp)


def mini_batch_quiz_1():
    m1 = np.random.rand(2, 3, 2)
    m2 = np.random.rand(1, 2, 3)
    print('dot:')
    print(np.dot(m1, m2).shape)
    print('matmul:')
    print(np.matmul(m1, m2).shape)


def mini_batch_quiz():
    data = np.loadtxt('data.csv', delimiter=',')
    X = data[:, :-1]
    print(np.shape(X))
    y = data[:, -1]
    print(y)
    print(np.shape(y))
    print("----")

    n_points = X.shape[0]
    W = np.zeros(X.shape[1])  # coefficients
    print(n_points)
    print(W)

    yhat = (W * X) + 1
    print(yhat)
    print(yhat.sum())


mini_batch_quiz_1()