import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_eager_execution()
n_values = 32
x= tf.linspace(-3.0 , 3.0 , n_values)

sess = tf.Session()
result = sess.run(x)

x.eval(session = sess)

sess.close()
sess = tf.InteractiveSession()

x.eval()

sigma = 1.0
mean = 0.0
z = (tf.exp(tf.negative(tf.pow(x-mean, 2.0) /
                        (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

assert z.graph is tf.get_default_graph()

plt.plot(z.eval())

print(z.get_shape())

print(z.get_shape().as_list())

print(tf.shape(z).eval())

print(tf.stack([tf.shape(z), tf.shape(z), [3], [4]]).eval())

z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))

# plt.imshow(z_2d.eval())

ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])


def gabor(n_values = 32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0)/
                            (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
    gauss_kernel = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.multiply(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel
plt.imshow(gabor().eval())

def convolve(img, W):
    if len(W.get_shape()) == 2:
        dims = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)

    if len(img.get_shape()) == 2:
        dims = [1] + W.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)

    elif len(img.get_shape()) ==3:
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        W = tf.concat(axis = 2, values=[W,W,W])

    convolved = tf.nn.conv2d(img, W, strides=[1,1,1,1], padding= 'SAME')

    return convolved

from skimage import data
img = data.astronaut()
# plt.imshow(img)
print(img.shape)

x= tf.placeholder(tf.float32, shape = img.shape)

out = convolve(x, gabor())

result = tf.squeeze(out).eval(feed_dict={x: img})
plt.imshow(result)






