import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const trainingDataPath = path.join(__dirname, '..', 'data', 'training');
const validationDataPath = path.join(__dirname, '..', 'data', 'validation');
const numClasses = fs.readdirSync(trainingDataPath).filter(subdir => fs.statSync(path.join(trainingDataPath, subdir)).isDirectory()).length;

async function preprocessImage(imagePath: string) {
    return tf.tidy(() => {
        const imageBuffer = fs.readFileSync(imagePath);
        return tf.node.decodePng(imageBuffer, 3)
            .resizeBilinear([128, 128])
            .div(255.0);
    });
}

async function loadData(dataPath: string) {
    const classDirs = fs.readdirSync(dataPath).filter(subdir => fs.statSync(path.join(dataPath, subdir)).isDirectory());
    const ds: { image: tf.Tensor, class: number }[] = [];

    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
        const classDir = classDirs[classIndex];
        const classPath = path.join(dataPath, classDir);
        const imageFiles = fs.readdirSync(classPath).filter(file => file.endsWith('.png'));

        for (let imageFile of imageFiles) {
            const imagePath = path.join(classPath, imageFile);
            const imageTensor = await preprocessImage(imagePath);
            ds.push({ image: imageTensor, class: classIndex });
        }
    }

    return ds;
}

async function shuffleData(ds: { image: tf.Tensor, class: number}[]) {
    tf.util.shuffle(ds);
    const { images, classes } = ds.reduce((acc, v) => {
        acc.images.push(v.image);
        acc.classes.push(v.class);
        return acc;
    }, { images: <tf.Tensor[]>[], classes: <number[]>[] });

    const xs = tf.stack(images);
    const ys = tf.oneHot(tf.tensor1d(classes, 'int32'), numClasses);

    return { xs, ys };
}

async function loadFeatureModel() {
    return tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v1_025_128/feature_vector/3/default/1', { fromTFHub: true });
}

async function createTransferModel(feature: any) {
    return tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [feature.shape[1]],
                units: 64,
                activation: 'relu',
            }),
            tf.layers.dense({ units: numClasses, activation: 'softmax' }),
        ],
    });
}

async function validateTransfer(featureModel: tf.GraphModel, transferModel: tf.Sequential) {
    const ds = await loadData(validationDataPath);
    const xs = tf.stack(ds.map((v) => v.image));
    const ys = tf.oneHot(tf.tensor1d(ds.map((v) => v.class), 'int32'), numClasses);

    const feature = await featureModel.predict(xs.reshape([ys.shape[0], 128, 128, 3])) as tf.Tensor;
    const result = transferModel.predict(feature) as tf.Tensor;

    const predictedClass = result.argMax(1).arraySync() as number[]; // This will give you an array of predicted class labels
    const validClass = ds.map((v) => v.class);

    const accuratePredicts = predictedClass.reduce((acc, v, i) => {
        if (v === validClass[i]) acc++;
        return acc;
    }, 0)
    const totalPredicts = predictedClass.length;

    console.log(`Accuracy: ${accuratePredicts / totalPredicts}`);
}

async function main() {
    const ds = await loadData(trainingDataPath);
    const { xs, ys } = await shuffleData(ds);

    const featureModel = await loadFeatureModel();
    const feature = featureModel.predict(xs.reshape([ys.shape[0], 128, 128, 3]));

    const transferModel = await createTransferModel(feature);
    transferModel.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    await transferModel.fit(feature, ys, {
        validationSplit: 0.2,
        epochs: 20,
    });

    await validateTransfer(featureModel, transferModel);
}

main().catch(console.error);
