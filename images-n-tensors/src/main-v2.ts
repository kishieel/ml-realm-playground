import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const imagePath = path.join(__dirname, '..', 'static', 'mask.jpg');
const binarizedPath = path.join(__dirname, '..', 'dist', 'binarized.jpg');


const imageBuffer = fs.readFileSync(imagePath);

tf.tidy(() => {
    const imageTensor = tf.node.decodeJpeg(imageBuffer, 1).div<tf.Tensor3D>(255.0);
    const thresholdTensor = imageTensor.greaterEqual(0.5);
    const binarizedTensor = tf.where(thresholdTensor, tf.ones(imageTensor.shape), tf.zeros(imageTensor.shape)).mul<tf.Tensor3D>(255.0);
    tf.node.encodeJpeg(binarizedTensor, 'grayscale').then((v) => fs.writeFileSync(binarizedPath, v));
});

console.log(tf.memory());
