import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const imagePath = path.join(__dirname, 'static', 'image.jpg');
const grayscalePath = path.join(__dirname, 'dist', 'grayscale.jpg');
const mirrorXPath = path.join(__dirname, 'dist', 'mirror-x.jpg');
const mirrorYPath = path.join(__dirname, 'dist', 'mirror-y.jpg');
const mirrorXYPath = path.join(__dirname, 'dist', 'mirror-xy.jpg');
const resizedBilinearPath = path.join(__dirname, 'dist', 'resized-bilinear.jpg');
const resizedNearestNeighborPath = path.join(__dirname, 'dist', 'resized-nearest-neighbor.jpg');
const randomRGBPath = path.join(__dirname, 'dist', 'random-rgb.jpg');
const randomGrayscalePath = path.join(__dirname, 'dist', 'random-grayscale.jpg');
const gradientGrayscalePath = path.join(__dirname, 'dist', 'gradient-grayscale.jpg');

const imageBuffer = fs.readFileSync(imagePath);

tf.tidy(() => {
    const imageTensor = tf.node.decodeJpeg(imageBuffer);

    const grayscaleTensor = imageTensor.mean<tf.Tensor3D>(-1, true);
    tf.node.encodeJpeg(grayscaleTensor).then((v) => fs.writeFileSync(grayscalePath, v));

    const mirrorXTensor = imageTensor.reverse([0]);
    tf.node.encodeJpeg(mirrorXTensor).then((v) => fs.writeFileSync(mirrorXPath, v));

    const mirrorYTensor = imageTensor.reverse([1]);
    tf.node.encodeJpeg(mirrorYTensor).then((v) => fs.writeFileSync(mirrorYPath, v));

    const mirrorXYTensor = imageTensor.reverse([0,1]);
    tf.node.encodeJpeg(mirrorXYTensor).then((v) => fs.writeFileSync(mirrorXYPath, v));

    const resizedBilinearTensor = imageTensor.resizeBilinear<tf.Tensor3D>([100,100]);
    tf.node.encodeJpeg(resizedBilinearTensor).then((v) => fs.writeFileSync(resizedBilinearPath, v));

    const resizedNearestNeighborTensor = imageTensor.resizeNearestNeighbor<tf.Tensor3D>([100,100]);
    tf.node.encodeJpeg(resizedNearestNeighborTensor).then((v) => fs.writeFileSync(resizedNearestNeighborPath, v));

    const randomRGBTensor = tf.randomUniform<tf.Rank.R3>([400,400,3], 0, 255);
    tf.node.encodeJpeg(randomRGBTensor).then((v) => fs.writeFileSync(randomRGBPath, v));

    const randomGrayscaleTensor = tf.randomUniform<tf.Rank.R3>([100,100,1], 0, 255);
    tf.node.encodeJpeg(randomGrayscaleTensor).then((v) => fs.writeFileSync(randomGrayscalePath, v));

    const gradientGrayscaleTensor = randomGrayscaleTensor.reshape([100,100]).topk(100).values.reshape<tf.Tensor3D>([100,100,1]);
    tf.node.encodeJpeg(gradientGrayscaleTensor).then((v) => fs.writeFileSync(gradientGrayscalePath, v));
});

console.log(tf.memory());
