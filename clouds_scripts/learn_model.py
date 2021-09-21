from sklearn import linear_model, model_selection, neural_network
import pandas as pd
import itertools

itertools.product()

dataset = pd.read_csv(r"E:\AI-58\clouds_databases\dataset.csv")

X = dataset[["f" + str(i)for i in range(108)]]
y = dataset["CM3up[W/m2]"]

X_train, X_validate, y_train, y_validate = model_selection.train_test_split(X, y, test_size=0.3)

# clf = linear_model.LinearRegression()
nn = neural_network.MLPRegressor(hidden_layer_sizes=[128, 32],
                                 verbose=True,
                                 max_iter=1000,
                                 )
nn.fit(X_train, y_train)

score = nn.score(X_validate, y_validate)
print(score)


