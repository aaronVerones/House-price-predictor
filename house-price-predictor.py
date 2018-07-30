# This tensorflow program predicts house price from square footage.
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation 


### STEP 1: Generate data ###

# Generate houses with a random house size.
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house )

# Generate house prices from house size. Add some random noise for more fun.
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)


### STEP 2: Prepare data ###

# Function to normalize values. prevents overflow or underflow.
def normalize(array):
    return (array - array.mean()) / array.std()

# Split the data into training and test data. 70% of it will be used for training, and 30% for testing.
num_train_samples = math.floor(num_house * 0.7)

# Use the first 70% as training data. That's ok, since the data is randomized.
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

# Use last 30% as testing data.
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

# Normalize.
train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)
test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)


### STEP 3: Set up TF containers ###

# Define TF placeholders for house size and price. These tensors are fed data at runtime.
# Multiple passes will be made through the data. Each pass, these placeholders are used to pass data to the optimizer algorithm.
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Define TF variables for the two values we're trying to solve for.
# The equation of a line is y = mx + b, where y is the price, and x is the square footage.
# In this example, m (slope) is size_factor, and b (intercept) is the price_offset.
# This will give us the equation of the line of best fit through our data.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")


### STEP 4: Set up learning model ###

# The prediction function is price = (size_factor * house_size) + price_offset. 
# Since this is tensor math, normal arithmetic functions cannot be used.
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# We will use Mean Squared Error to determine the strength of our fit. 
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

# We will use Gradient Descent to optimize our model, with a step of 0.1.
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)


### STEP 5: Launch training session ###

# Initialize TF global variables.
init = tf.global_variables_initializer()

# Launch the TF execution graph.
with tf.Session() as sess:
    sess.run(init)

    # We'll do 50 iterations and display the progress every 2 iterations.
    display_every = 2
    num_training_iter = 50

    # This is for a neat animation that shows the model's progress graphically.
    fit_num_plots = math.floor(num_training_iter/display_every)
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0    

   # Iterate over the training data num_training_iter times.
    for iteration in range(num_training_iter):

        # Run the optimizer. This is the meat and potatoes of the learning.
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # Display current status in the command line
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost =", "{:.9f}".format(c), \
                "size_factor =", sess.run(tf_size_factor), "price_offset =", sess.run(tf_price_offset))
            # Store the current size_factor and price_offset for the progress animation
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1

    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost =", training_cost, "size_factor =", sess.run(tf_size_factor), "price_offset =", sess.run(tf_price_offset), '\n')

  
    # Show a plot that renders our line of best fit as the Gradient Descent optimization progresses.

    # We normalized the data earlier, but in order to show it properly now, we have to denormalize it.
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()
    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams["figure.figsize"] = (10,8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)
        return line,
 
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0.
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=200, blit=True)

    plt.show()   
