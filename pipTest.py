from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 50
pd.options.display.float_format = '{:.1f}'.format
# 读取数据集
california_housing_dataframe = pd.read_csv("D:\TestData\california_housing_train.csv")
# 对数据集里面的数据进行重新排序
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
# 降低数字大小，提高学习效率
california_housing_dataframe['median_house_value'] /= 1000.0
my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
targets = california_housing_dataframe["median_house_value"]
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


# 定义一个输入函数，对数据进行预处理。
def my_input_fn(features, targets, batch_size=5, shuffle=True, num_epochs=None):
    # 将Csv数据转化成一个np数组。np数组能够保证一个list中全部都是一个类型的数据。
    # 接着将数据切片，不然输入数据是整个数据集
    # 切片后再进行分批处理。此处size为1,代表每次输入一个数据
    # repeat是指数据要被重复多少次。暂时不太理解重复的意义，可能和数据处理有关。
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    # 将数据随机打乱。
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 由于这是进行预测的数据，所以不打乱顺序，输入的时候不需要重复，一次只输入一个数据。

# #用训练集的数据进行测试
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# predictions = np.array([item['predictions'][0] for item in predictions])
#
# mean_squared_error = metrics.mean_squared_error(predictions,targets)
# root_mean_squared_error = math.sqrt(mean_squared_error)
# print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
# print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title("Learned Line by Peroid")
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")
# 去样本点点在图上，营造一种这条线确实是用线性回归做出来的氛围。
sample = california_housing_dataframe.sample(n=300)
plt.scatter(sample["total_rooms"], sample["median_house_value"])
#colors = [cm. for x in np.linspace(-1, 1, 10)]
root_mean_squared_errors = []
for peroid in range(0, 10):
    linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=200
    )
    prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    # 计算均方根误差
    root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
    print(" period %02d : %0.2f" % (peroid,root_mean_squared_error))
    root_mean_squared_errors.append(root_mean_squared_error)
    #确定x、y轴的末端值
    y_extents = np.array([0, sample["median_house_value"].max()])
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % "total_rooms")[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample["total_rooms"].max()),
                           sample["total_rooms"].min())
    plt.plot(x_extents,y_extents)

plt.show()
print("It's all over.")

