import * as tf from '@tensorflow/tfjs'

const x1 = tf.tensor([
    [3, 9, 3, 1],
    [1, 42, 12, 1],
    [12, 3, 24, 50],
]);

const y1 = tf.tensor([
    [4, 0, 2],
    [3, 32, 0],
    [0, 0, 23],
    [12, 6, 7],
]);

const xy1_mul = tf.mul(x1.slice([0,0], [3,3]), y1.slice([0,0], [3,3]));
xy1_mul.print();

const xy1_matMul = tf.matMul(x1, y1);
xy1_matMul.print();

const xy1_dot = tf.dot(x1, y1);
xy1_dot.print();

